/*
 * Copyright 2023-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.gemini.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.ai.gemini.api.common.GeminiApiConstants;
import org.springframework.ai.model.ApiKey;
import org.springframework.ai.model.ChatModelDescription;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.SimpleApiKey;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.ResponseErrorHandler;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Single class implementation of the
 * <a href="https://ai.google.dev/gemini-api/docs/text-generation">Gemini Text Generation
 * API</a> and <a href="https://ai.google.dev/gemini-api/docs/embeddings">Gemini
 * Embedding API</a>.
 *
 * @author Jonghoon Park
 */
public class GeminiApi {

	public static Builder builder() {
		return new Builder();
	}

	public static final GeminiApi.ChatModel DEFAULT_CHAT_MODEL = ChatModel.GEMINI_2_O_FLASH;

	public static final String DEFAULT_EMBEDDING_MODEL = EmbeddingModel.GEMINI_EMBEDDING_EXP_03_07.getValue();

	private static final Predicate<String> SSE_DONE_PREDICATE = "[DONE]"::equals;

	private final ApiKey apiKey;

	private final String completionsPath;

	private final String embeddingsPath;

	private final RestClient restClient;

	private final WebClient webClient;

	/**
	 * Create a new chat completion api.
	 * @param baseUrl api base URL.
	 * @param apiKey Gemini apiKey.
	 * @param headers the http headers to use.
	 * @param completionsPath the path to the chat completions endpoint.
	 * @param embeddingsPath the path to the embeddings endpoint.
	 * @param restClientBuilder RestClient builder.
	 * @param webClientBuilder WebClient builder.
	 * @param responseErrorHandler Response error handler.
	 */
	public GeminiApi(String baseUrl, ApiKey apiKey, MultiValueMap<String, String> headers, String completionsPath,
					 String embeddingsPath, RestClient.Builder restClientBuilder, WebClient.Builder webClientBuilder,
					 ResponseErrorHandler responseErrorHandler) {

		Assert.hasText(completionsPath, "Completions Path must not be null");
		Assert.hasText(embeddingsPath, "Embeddings Path must not be null");
		Assert.notNull(headers, "Headers must not be null");

		this.apiKey = apiKey;
		this.completionsPath = completionsPath;
		this.embeddingsPath = embeddingsPath;
		// @formatter:off
		Consumer<HttpHeaders> finalHeaders = h -> {
			h.setContentType(MediaType.APPLICATION_JSON);
			h.addAll(headers);
		};
		this.restClient = restClientBuilder.baseUrl(baseUrl)
			.defaultHeaders(finalHeaders)
			.defaultStatusHandler(responseErrorHandler)
			.build();

		this.webClient = webClientBuilder
			.baseUrl(baseUrl)
			.defaultHeaders(finalHeaders)
			.build(); // @formatter:on
	}

	/**
	 * Creates a model response for the given chat conversation.
	 * @param chatRequest The chat completion request.
	 * @return Entity response with {@link ChatCompletion} as a body and HTTP status code
	 * and headers.
	 */
	public ResponseEntity<ChatCompletion> chatCompletionEntity(ChatCompletionRequest chatRequest) {
		return chatCompletionEntity(chatRequest, new LinkedMultiValueMap<>());
	}

	/**
	 * Creates a model response for the given chat conversation.
	 * @param chatRequest The chat completion request.
	 * @param additionalHttpHeader Optional, additional HTTP headers to be added to the
	 * request.
	 * @return Entity response with {@link ChatCompletion} as a body and HTTP status code
	 * and headers.
	 */
	public ResponseEntity<ChatCompletion> chatCompletionEntity(ChatCompletionRequest chatRequest,
															   MultiValueMap<String, String> additionalHttpHeader) {

		Assert.notNull(chatRequest, "The request can not be null.");
		Assert.isTrue(!chatRequest.stream(), "Request must set the stream property to false.");
		Assert.notNull(additionalHttpHeader, "The additional HTTP headers can not be null.");

		return this.restClient.post()
			.uri(String.format("%s/%s:generateContent?key=%s", this.completionsPath, chatRequest.model, this.apiKey.getValue()))
			.headers(headers -> headers.addAll(additionalHttpHeader))
			.body(new Message(chatRequest.contents))
			.retrieve()
			.toEntity(ChatCompletion.class);
	}

	/**
	 * Creates a streaming chat response for the given chat conversation.
	 * @param generationRequest The chat completion request. Must have the stream property set
	 * to true.
	 * @return Returns a {@link Flux} stream from chat completion chunks.
	 */
	public Flux<ChatCompletion> chatCompletionStream(ChatCompletionRequest generationRequest) {
		return chatCompletionStream(generationRequest, new LinkedMultiValueMap<>());
	}

	/**
	 * Creates a streaming chat response for the given chat conversation.
	 * @param chatRequest The chat completion request. Must have the stream property set
	 * to true.
	 * @param additionalHttpHeader Optional, additional HTTP headers to be added to the
	 * request.
	 * @return Returns a {@link Flux} stream from chat completion chunks.
	 */
	public Flux<ChatCompletion> chatCompletionStream(ChatCompletionRequest chatRequest,
													 MultiValueMap<String, String> additionalHttpHeader) {

		Assert.notNull(chatRequest, "The request body can not be null.");
		Assert.isTrue(chatRequest.stream(), "Request must set the stream property to true.");

		AtomicBoolean isInsideTool = new AtomicBoolean(false);

		return this.webClient.post()
				.uri(String.format("%s/%s:streamGenerateContent?alt=sse&key=%s", this.completionsPath, chatRequest.model, this.apiKey.getValue()))
				.headers(headers -> headers.addAll(additionalHttpHeader))
				.body(Mono.just(new Message(chatRequest.contents)), Message.class)
				.retrieve()
				.bodyToFlux(String.class)
				// cancels the flux stream after the "[DONE]" is received.
				.takeUntil(SSE_DONE_PREDICATE)
				// filters out the "[DONE]" content.
				.filter(SSE_DONE_PREDICATE.negate())
				.map(content -> ModelOptionsUtils.jsonToObject(content, ChatCompletion.class))
				// Group all chunks belonging to the same function call.
				// Flux<ChatCompletionChunk> -> Flux<Flux<ChatCompletionChunk>>
				.windowUntil(chunk -> {
//					if (isInsideTool.get() && this.chunkMerger.isStreamingToolFunctionCallFinish(chunk)) {
//						isInsideTool.set(false);
//						return true;
//					}
					return !isInsideTool.get();
				})
				// Merging the window chunks into a single chunk.
				// Reduce the inner Flux<ChatCompletionChunk> window into a single
				// Mono<ChatCompletionChunk>,
				// Flux<Flux<ChatCompletionChunk>> -> Flux<Mono<ChatCompletionChunk>>
//				.concatMapIterable(window -> {
//					Mono<ChatCompletionChunk> monoChunk = window.reduce(
//							new ChatCompletionChunk(null, null, null, null, null, null, null, null),
//							(previous, current) -> this.chunkMerger.merge(previous, current));
//					return List.of(monoChunk);
//				})
				// Flux<Mono<ChatCompletionChunk>> -> Flux<ChatCompletionChunk>
				.flatMap(mono -> mono);
	}


	/**
	 * Creates an embedding vector representing the input text or token array.
	 * @param embeddingRequest The embedding request.
	 * @return Returns list of {@link Embedding} wrapped in {@link EmbeddingList}.
	 * @param <T> Type of the entity in the data list. Can be a {@link String} or
	 * {@link List} of tokens (e.g. Integers). For embedding multiple inputs in a single
	 * request, You can pass a {@link List} of {@link String} or {@link List} of
	 * {@link List} of tokens. For example:
	 *
	 * <pre>{@code List.of("text1", "text2", "text3") or List.of(List.of(1, 2, 3), List.of(3, 4, 5))} </pre>
	 */
	public <T> ResponseEntity<EmbeddingList<Embedding>> embeddings(EmbeddingRequest<T> embeddingRequest) {

		Assert.notNull(embeddingRequest, "The request body can not be null.");

		// Input text to embed, encoded as a string or array of tokens. To embed multiple
		// inputs in a single
		// request, pass an array of strings or array of token arrays.
		Assert.notNull(embeddingRequest.input(), "The input can not be null.");
		Assert.isTrue(embeddingRequest.input() instanceof String || embeddingRequest.input() instanceof List,
				"The input must be either a String, or a List of Strings or List of List of integers.");

		// The input must not exceed the max input tokens for the model (8192 tokens for
		// text-embedding-ada-002), cannot
		// be an empty string, and any array must be 2048 dimensions or less.
		if (embeddingRequest.input() instanceof List list) {
			Assert.isTrue(!CollectionUtils.isEmpty(list), "The input list can not be empty.");
			Assert.isTrue(list.size() <= 2048, "The list must be 2048 dimensions or less");
			Assert.isTrue(
					list.get(0) instanceof String || list.get(0) instanceof Integer || list.get(0) instanceof List,
					"The input must be either a String, or a List of Strings or list of list of integers.");
		}

		return this.restClient.post()
			.uri(this.embeddingsPath)
			.body(embeddingRequest)
			.retrieve()
			.toEntity(new ParameterizedTypeReference<>() {

			});
	}

	/**
	 * Gemini Chat Completion Models.
	 * <p>
	 * This enum provides a selective list of chat completion models available through the
	 * Gemini API, along with their key features and links to the official Gemini
	 * documentation for further details.
	 * <p>
	 * The models are grouped by their capabilities and intended use cases. For each
	 * model, a brief description is provided, highlighting its strengths, limitations,
	 * and any specific features. When available, the description also includes
	 * information about the model's context window, maximum output tokens, and knowledge
	 * cutoff date.
	 * <p>
	 * <b>References:</b>
	 * <ul>
	 * <li><a href="https://platform.openai.com/docs/models#gpt-4o">GPT-4o</a></li>
	 * <li><a href="https://platform.openai.com/docs/models#gpt-4-and-gpt-4-turbo">GPT-4
	 * and GPT-4 Turbo</a></li>
	 * <li><a href="https://platform.openai.com/docs/models#gpt-3-5-turbo">GPT-3.5
	 * Turbo</a></li>
	 * <li><a href="https://platform.openai.com/docs/models#o1-and-o1-mini">o1 and
	 * o1-mini</a></li>
	 * <li><a href="https://platform.openai.com/docs/models#o3-mini">o3-mini</a></li>
	 * </ul>
	 */
	public enum ChatModel implements ChatModelDescription {

		/**
		 * TODO: update model description
		 */
		GEMINI_2_O_FLASH("gemini-2.0-flash");

		/**
		 * TODO: add other models
		 */

		public final String value;

		ChatModel(String value) {
			this.value = value;
		}

		public String getValue() {
			return this.value;
		}

		@Override
		public String getName() {
			return this.value;
		}

	}

	/**
	 * The reason the model stopped generating tokens.
	 */
	public enum ChatCompletionFinishReason {

		/**
		 * The model hit a natural stop point or a provided stop sequence.
		 */
		@JsonProperty("stop")
		STOP,
		/**
		 * The maximum number of tokens specified in the request was reached.
		 */
		@JsonProperty("length")
		LENGTH,
		/**
		 * The content was omitted due to a flag from our content filters.
		 */
		@JsonProperty("content_filter")
		CONTENT_FILTER,
		/**
		 * The model called a tool.
		 */
		@JsonProperty("tool_calls")
		TOOL_CALLS,
		/**
		 * Only for compatibility with Mistral AI API.
		 */
		@JsonProperty("tool_call")
		TOOL_CALL

	}

	public enum Modality {
		MODALITY_UNSPECIFIED("MODALITY_UNSPECIFIED"),
		TEXT("TEXT"),
		IMAGE("IMAGE"),
		AUDIO("AUDIO");

		public final String value;

		Modality(String value) {
			this.value = value;
		}
	}

	/**
	 * Gemini Embeddings Models:
	 * <a href="https://platform.openai.com/docs/models/embeddings">Embeddings</a>.
	 */
	public enum EmbeddingModel {

		/**
		 * TODO: update model description
		 */
		GEMINI_EMBEDDING_EXP_03_07("gemini-embedding-exp-03-07"),
		/**
		 * TODO: update model description
		 */
		TEXT_EMBEDDING_004("text-embedding-004");

		public final String value;

		EmbeddingModel(String value) {
			this.value = value;
		}

		public String getValue() {
			return this.value;
		}

	}

	/**
	 * Creates a model response for the given chat conversation.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ChatCompletionRequest(// @formatter:off
			@JsonProperty("contents") List<Content> contents,
			@JsonProperty("model") String model,
			@JsonProperty("temperature") Double temperature,
			@JsonProperty("stream") Boolean stream,
			@JsonProperty("top_p") Double topP,
			@JsonProperty("top_k") Double topK) {

		/**
		 * Shortcut constructor for a chat completion request with the given messages, model and temperature.
		 *
		 * @param messages A list of messages comprising the conversation so far.
		 * @param model ID of the model to use.
		 * @param temperature What sampling temperature to use, between 0 and 1.
		 */
		public ChatCompletionRequest(List<Content> messages, String model, Double temperature) {
//			this(messages, model, null, null, null, null, null, null, null, null, null, null, null, null, null,
//					null, null, null, false, null, temperature, null,
//					null, null, null, null, null);
			this(messages, model, temperature, null, null, null);
		}

		/**
		 * Shortcut constructor for a chat completion request with the given messages, model, temperature and control for streaming.
		 *
		 * @param messages A list of messages comprising the conversation so far.
		 * @param model ID of the model to use.
		 * @param temperature What sampling temperature to use, between 0 and 1.
		 * @param stream If set, partial content deltas will be sent.Tokens will be sent as data-only server-sent events
		 * as they become available, with the stream terminated by a data: [DONE] content.
		 */
		public ChatCompletionRequest(List<Content> messages, String model, Double temperature, boolean stream) {
			/*this(messages, model, null, null, null, null, null, null, null, null, null,
					null, null, null, null, null, stream, null, temperature, null,
					null, null, null);*/
			this(messages, model, temperature, stream, null, null);
		}

		/**
		 * Shortcut constructor for a chat completion request with the given messages for streaming.
		 *
		 * @param messages A list of messages comprising the conversation so far.
		 * @param stream If set, partial content deltas will be sent.Tokens will be sent as data-only server-sent events
		 * as they become available, with the stream terminated by a data: [DONE] content.
		 */
		public ChatCompletionRequest(List<Content> messages, Boolean stream) {
			this(messages, null, null, stream, null, null);
		}
	} // @formatter:on

	@JsonInclude(Include.NON_NULL)
	public record Message(
			@JsonProperty("contents") List<Content> contents
	) {

	}

	/**
	 * Message comprising the conversation.
	 */
	@JsonInclude(Include.NON_NULL)
	public record Content(// @formatter:off
			@JsonProperty("parts") List<Part> parts,
			@JsonProperty("role") Role role) { // @formatter:on

//		/**
//		 * Get content content as String.
//		 */
//		public String content() {
//			if (this.rawContent == null) {
//				return null;
//			}
//			if (this.rawContent instanceof String text) {
//				return text;
//			}
//			throw new IllegalStateException("The content is not a string!");
//		}

		/**
		 * The role of the author of this content.
		 */
		public enum Role {

			/**
			 * User content.
			 */
			@JsonProperty("user")
			USER,
			/**
			 * Assistant content.
			 */
			@JsonProperty("model")
			MODEL

		}

		public record Part(@JsonProperty("text") String text) {

		}
	}

	/**
	 * Represents a chat completion response returned by model, based on the provided
	 * input.
	 *
	 * @param id A unique identifier for the chat completion.
	 * @param created The Unix timestamp (in seconds) of when the chat completion was
	 * created.
	 * @param model The model used for the chat completion.
	 * @param object The object type, which is always chat.completion.
	 * @param usage Usage statistics for the completion request.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ChatCompletion(// @formatter:off
		  	List<Candidate> candidates,

//		  PromptFeedback promptFeedback,
			@JsonProperty("usageMetadata") Usage usage,
		    String modelVersion
	) { // @formatter:on

	}

	@JsonInclude(Include.NON_NULL)
	public record Candidate(// @formatter:off
							@JsonProperty("finish_reason") ChatCompletionFinishReason finishReason,
							@JsonProperty("content") Content content) { // @formatter:on
	}
	// Embeddings API

	/**
	 * Usage statistics for the completion request.
	 *
	 * @param completionTokens Number of tokens in the generated completion. Only
	 * applicable for completion requests.
	 * @param promptTokens Number of tokens in the prompt.
	 * @param totalTokens Total number of tokens used in the request (prompt +
	 * completion).
	 * @param promptTokensDetails Breakdown of tokens used in the prompt.
	 * @param completionTokenDetails Breakdown of tokens used in a completion.
	 * @param promptCacheHitTokens Number of tokens in the prompt that were served from
	 * (util for
	 * <a href="https://api-docs.deepseek.com/api/create-chat-completion">DeepSeek</a>
	 * support).
	 * @param promptCacheMissTokens Number of tokens in the prompt that were not served
	 * (util for
	 * <a href="https://api-docs.deepseek.com/api/create-chat-completion">DeepSeek</a>
	 * support).
	 */
	@JsonInclude(Include.NON_NULL)
	@JsonIgnoreProperties(ignoreUnknown = true)
	public record Usage(// @formatter:off
		Integer promptTokenCount,
		Integer cachedContentTokenCount,
		Integer candidatesTokenCount,
		Integer toolUsePromptTokenCount,
		Integer thoughtsTokenCount,
		Integer totalTokenCount,
		List<ModalityTokenCount> promptTokensDetails,
		List<ModalityTokenCount> cacheTokensDetails,
		List<ModalityTokenCount> candidatesTokensDetails,
		List<ModalityTokenCount> toolUsePromptTokensDetails) { // @formatter:on

		/**
		 * Breakdown of tokens used in the prompt
		 *
		 * @param audioTokens Audio input tokens present in the prompt.
		 * @param cachedTokens Cached tokens present in the prompt.
		 */
		@JsonInclude(Include.NON_NULL)
		public record ModalityTokenCount(// @formatter:off
			Modality modality,
			Integer tokenCount) { // @formatter:on
		}

		/**
		 * Breakdown of tokens used in a completion.
		 *
		 * @param reasoningTokens Number of tokens generated by the model for reasoning.
		 * @param acceptedPredictionTokens Number of tokens generated by the model for
		 * accepted predictions.
		 * @param audioTokens Number of tokens generated by the model for audio.
		 * @param rejectedPredictionTokens Number of tokens generated by the model for
		 * rejected predictions.
		 */
		@JsonInclude(Include.NON_NULL)
		@JsonIgnoreProperties(ignoreUnknown = true)
		public record CompletionTokenDetails(// @formatter:off
			@JsonProperty("reasoning_tokens") Integer reasoningTokens,
			@JsonProperty("accepted_prediction_tokens") Integer acceptedPredictionTokens,
			@JsonProperty("audio_tokens") Integer audioTokens,
			@JsonProperty("rejected_prediction_tokens") Integer rejectedPredictionTokens) { // @formatter:on
		}
	}



	/**
	 * Represents a streamed chunk of a chat completion response returned by model, based
	 * on the provided input.
	 *
	 * @param id A unique identifier for the chat completion. Each chunk has the same ID.
	 * @param choices A list of chat completion choices. Can be more than one if n is
	 * greater than 1.
	 * @param created The Unix timestamp (in seconds) of when the chat completion was
	 * created. Each chunk has the same timestamp.
	 * @param model The model used for the chat completion.
	 * @param serviceTier The service tier used for processing the request. This field is
	 * only included if the service_tier parameter is specified in the request.
	 * @param systemFingerprint This fingerprint represents the backend configuration that
	 * the model runs with. Can be used in conjunction with the seed request parameter to
	 * understand when backend changes have been made that might impact determinism.
	 * @param object The object type, which is always 'chat.completion.chunk'.
	 * @param usage Usage statistics for the completion request. Present in the last chunk
	 * only if the StreamOptions.includeUsage is set to true.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ChatCompletionChunk(// @formatter:off
			@JsonProperty("id") String id,
			@JsonProperty("candidates") List<Candidate> candidates,
			@JsonProperty("created") Long created,
			@JsonProperty("model") String model,
			@JsonProperty("service_tier") String serviceTier,
			@JsonProperty("system_fingerprint") String systemFingerprint,
			@JsonProperty("object") String object,
			@JsonProperty("usageMetaData") Usage usage) { // @formatter:on

	}

	/**
	 * Log probability information for the choice.
	 *
	 * @param content A list of content content tokens with log probability information.
	 * @param refusal A list of content refusal tokens with log probability information.
	 */
	@JsonInclude(Include.NON_NULL)
	public record LogProbs(@JsonProperty("content") List<Content> content,
						   @JsonProperty("refusal") List<Content> refusal) {

		/**
		 * Message content tokens with log probability information.
		 *
		 * @param token The token.
		 * @param logprob The log probability of the token.
		 * @param probBytes A list of integers representing the UTF-8 bytes representation
		 * of the token. Useful in instances where characters are represented by multiple
		 * tokens and their byte representations must be combined to generate the correct
		 * text representation. Can be null if there is no bytes representation for the
		 * token.
		 * @param topLogprobs List of the most likely tokens and their log probability, at
		 * this token position. In rare cases, there may be fewer than the number of
		 * requested top_logprobs returned.
		 */
		@JsonInclude(Include.NON_NULL)
		public record Content(// @formatter:off
							  @JsonProperty("token") String token,
							  @JsonProperty("logprob") Float logprob,
							  @JsonProperty("bytes") List<Integer> probBytes,
							  @JsonProperty("top_logprobs") List<TopLogProbs> topLogprobs) { // @formatter:on

			/**
			 * The most likely tokens and their log probability, at this token position.
			 *
			 * @param token The token.
			 * @param logprob The log probability of the token.
			 * @param probBytes A list of integers representing the UTF-8 bytes
			 * representation of the token. Useful in instances where characters are
			 * represented by multiple tokens and their byte representations must be
			 * combined to generate the correct text representation. Can be null if there
			 * is no bytes representation for the token.
			 */
			@JsonInclude(Include.NON_NULL)
			public record TopLogProbs(// @formatter:off
									  @JsonProperty("token") String token,
									  @JsonProperty("logprob") Float logprob,
									  @JsonProperty("bytes") List<Integer> probBytes) { // @formatter:on
			}

		}

	}

	/**
	 * Represents an embedding vector returned by embedding endpoint.
	 *
	 * @param index The index of the embedding in the list of embeddings.
	 * @param embedding The embedding vector, which is a list of floats. The length of
	 * vector depends on the model.
	 * @param object The object type, which is always 'embedding'.
	 */
	@JsonInclude(Include.NON_NULL)
	public record Embedding(// @formatter:off
			@JsonProperty("index") Integer index,
			@JsonProperty("embedding") float[] embedding,
			@JsonProperty("object") String object) { // @formatter:on

		/**
		 * Create an embedding with the given index, embedding and object type set to
		 * 'embedding'.
		 * @param index The index of the embedding in the list of embeddings.
		 * @param embedding The embedding vector, which is a list of floats. The length of
		 * vector depends on the model.
		 */
		public Embedding(Integer index, float[] embedding) {
			this(index, embedding, "embedding");
		}

	}

	/**
	 * Creates an embedding vector representing the input text.
	 *
	 * @param <T> Type of the input.
	 * @param input Input text to embed, encoded as a string or array of tokens. To embed
	 * multiple inputs in a single request, pass an array of strings or array of token
	 * arrays. The input must not exceed the max input tokens for the model (8192 tokens
	 * for text-embedding-ada-002), cannot be an empty string, and any array must be 2048
	 * dimensions or less.
	 * @param model ID of the model to use.
	 * @param encodingFormat The format to return the embeddings in. Can be either float
	 * or base64.
	 * @param dimensions The number of dimensions the resulting output embeddings should
	 * have. Only supported in text-embedding-3 and later models.
	 * @param user A unique identifier representing your end-user, which can help Gemini
	 * to monitor and detect abuse.
	 */
	@JsonInclude(Include.NON_NULL)
	public record EmbeddingRequest<T>(// @formatter:off
			@JsonProperty("input") T input,
			@JsonProperty("model") String model,
			@JsonProperty("encoding_format") String encodingFormat,
			@JsonProperty("dimensions") Integer dimensions,
			@JsonProperty("user") String user) { // @formatter:on

		/**
		 * Create an embedding request with the given input, model and encoding format set
		 * to float.
		 * @param input Input text to embed.
		 * @param model ID of the model to use.
		 */
		public EmbeddingRequest(T input, String model) {
			this(input, model, "float", null, null);
		}

		/**
		 * Create an embedding request with the given input. Encoding format is set to
		 * float and user is null and the model is set to 'text-embedding-ada-002'.
		 * @param input Input text to embed.
		 */
		public EmbeddingRequest(T input) {
			this(input, DEFAULT_EMBEDDING_MODEL);
		}

	}

	/**
	 * List of multiple embedding responses.
	 *
	 * @param <T> Type of the entities in the data list.
	 * @param object Must have value "list".
	 * @param data List of entities.
	 * @param model ID of the model to use.
	 * @param usage Usage statistics for the completion request.
	 */
	@JsonInclude(Include.NON_NULL)
	public record EmbeddingList<T>(// @formatter:off
			@JsonProperty("object") String object,
			@JsonProperty("data") List<T> data,
			@JsonProperty("model") String model,
			@JsonProperty("usageMetaData") Usage usage) { // @formatter:on
	}

	public static class Builder {

		private String baseUrl = GeminiApiConstants.DEFAULT_BASE_URL;

		private ApiKey apiKey;

		private MultiValueMap<String, String> headers = new LinkedMultiValueMap<>();

		private String completionsPath = "/v1beta/models/";

		private String embeddingsPath = "/v1beta/models/";

		private RestClient.Builder restClientBuilder = RestClient.builder();

		private WebClient.Builder webClientBuilder = WebClient.builder();

		private ResponseErrorHandler responseErrorHandler = RetryUtils.DEFAULT_RESPONSE_ERROR_HANDLER;

		public Builder baseUrl(String baseUrl) {
			Assert.hasText(baseUrl, "baseUrl cannot be null or empty");
			this.baseUrl = baseUrl;
			return this;
		}

		public Builder apiKey(ApiKey apiKey) {
			Assert.notNull(apiKey, "apiKey cannot be null");
			this.apiKey = apiKey;
			return this;
		}

		public Builder apiKey(String simpleApiKey) {
			Assert.notNull(simpleApiKey, "simpleApiKey cannot be null");
			this.apiKey = new SimpleApiKey(simpleApiKey);
			return this;
		}

		public Builder headers(MultiValueMap<String, String> headers) {
			Assert.notNull(headers, "headers cannot be null");
			this.headers = headers;
			return this;
		}

		public Builder completionsPath(String completionsPath) {
			Assert.hasText(completionsPath, "completionsPath cannot be null or empty");
			this.completionsPath = completionsPath;
			return this;
		}

		public Builder embeddingsPath(String embeddingsPath) {
			Assert.hasText(embeddingsPath, "embeddingsPath cannot be null or empty");
			this.embeddingsPath = embeddingsPath;
			return this;
		}

		public Builder restClientBuilder(RestClient.Builder restClientBuilder) {
			Assert.notNull(restClientBuilder, "restClientBuilder cannot be null");
			this.restClientBuilder = restClientBuilder;
			return this;
		}

		public Builder webClientBuilder(WebClient.Builder webClientBuilder) {
			Assert.notNull(webClientBuilder, "webClientBuilder cannot be null");
			this.webClientBuilder = webClientBuilder;
			return this;
		}

		public Builder responseErrorHandler(ResponseErrorHandler responseErrorHandler) {
			Assert.notNull(responseErrorHandler, "responseErrorHandler cannot be null");
			this.responseErrorHandler = responseErrorHandler;
			return this;
		}

		public GeminiApi build() {
			Assert.notNull(this.apiKey, "apiKey must be set");
			return new GeminiApi(this.baseUrl, this.apiKey, this.headers, this.completionsPath, this.embeddingsPath,
					this.restClientBuilder, this.webClientBuilder, this.responseErrorHandler);
		}

	}

}
