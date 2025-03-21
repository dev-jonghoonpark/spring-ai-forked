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

package org.springframework.ai.gemini;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import io.micrometer.observation.ObservationRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.metadata.Usage;
import reactor.core.publisher.Flux;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.metadata.EmptyUsage;
import org.springframework.ai.chat.metadata.RateLimit;
import org.springframework.ai.chat.metadata.UsageUtils;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.MessageAggregator;
import org.springframework.ai.chat.model.StreamingChatModel;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.chat.observation.DefaultChatModelObservationConvention;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.Media;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.gemini.api.GeminiApi;
import org.springframework.ai.gemini.api.GeminiApi.ChatCompletionChunk;
import org.springframework.ai.gemini.api.GeminiApi.ChatCompletion;
import org.springframework.ai.gemini.api.GeminiApi.Candidate;
import org.springframework.ai.gemini.api.GeminiApi.Content;
import org.springframework.ai.gemini.api.GeminiApi.ChatCompletionRequest;
import org.springframework.ai.gemini.metadata.support.GeminiResponseHeaderExtractor;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.MultiValueMap;

/**
 * {@link ChatModel} and {@link StreamingChatModel} implementation for {@literal Gemini}
 * backed by {@link GeminiApi}.
 *
 * @author Jonghoon Park
 * @see ChatModel
 * @see StreamingChatModel
 * @see GeminiApi
 */
public class GeminiChatModel implements ChatModel {

	private static final Logger logger = LoggerFactory.getLogger(GeminiChatModel.class);

	private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultChatModelObservationConvention();

	/**
	 * The default options used for the chat completion requests.
	 */
	private final GeminiChatOptions defaultOptions;

	/**
	 * The retry template used to retry the Gemini API calls.
	 */
	private final RetryTemplate retryTemplate;

	/**
	 * Low-level access to the Gemini API.
	 */
	private final GeminiApi geminiApi;

	/**
	 * Conventions to use for generating observations.
	 */
	private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	public GeminiChatModel(GeminiApi geminiApi, GeminiChatOptions defaultOptions,
						   RetryTemplate retryTemplate) {
		Assert.notNull(geminiApi, "geminiApi cannot be null");
		Assert.notNull(defaultOptions, "defaultOptions cannot be null");
		Assert.notNull(retryTemplate, "retryTemplate cannot be null");
		this.geminiApi = geminiApi;
		this.defaultOptions = defaultOptions;
		this.retryTemplate = retryTemplate;
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		// Before moving any further, build the final request Prompt,
		// merging runtime and default options.
		Prompt requestPrompt = buildRequestPrompt(prompt);
		return this.internalCall(requestPrompt, null);
	}

	public ChatResponse internalCall(Prompt prompt, ChatResponse previousChatResponse) {

		ChatCompletionRequest request = createRequest(prompt, false);

		ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
			.prompt(prompt)
			.provider(org.springframework.ai.gemini.api.common.GeminiApiConstants.PROVIDER_NAME)
			.requestOptions(prompt.getOptions())
			.build();

		ResponseEntity<ChatCompletion> completionEntity = this.retryTemplate
			.execute(ctx -> this.geminiApi.chatCompletionEntity(request, getAdditionalHttpHeaders(prompt)));

		var chatCompletion = completionEntity.getBody();

//		List<Choice> choices = chatCompletion.choices();
//		if (choices == null) {
//			logger.warn("No choices returned for prompt: {}", prompt);
//			return new ChatResponse(List.of());
//		}

		List<Candidate> candidates = chatCompletion.candidates();
		if (candidates == null) {
			logger.warn("No candidates returned for prompt: {}", prompt);
			return new ChatResponse(List.of());
		}

		List<Generation> generations = candidates.stream().map(candidate -> {
			// @formatter:off
			Map<String, Object> metadata = Map.of(
					"role", candidate.content().role() != null ? candidate.content().role().name() : ""
//					"index", candidate.index(),
//					"finishReason", candidate.finishReason() != null ? candidate.finishReason().name() : ""
			);
			// @formatter:on
			return buildGeneration(candidate, metadata, request);
		}).toList();

		RateLimit rateLimit = GeminiResponseHeaderExtractor.extractAiResponseHeaders(completionEntity);

		// Current usage
		GeminiApi.Usage usage = completionEntity.getBody().usage();
		Usage currentChatResponseUsage = usage != null ? getDefaultUsage(usage) : new EmptyUsage();
		Usage accumulatedUsage = UsageUtils.getCumulativeUsage(currentChatResponseUsage, previousChatResponse);
		ChatResponse chatResponse = new ChatResponse(generations,
				from(completionEntity.getBody(), rateLimit, accumulatedUsage));

		observationContext.setResponse(chatResponse);

		return chatResponse;
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {
		// Before moving any further, build the final request Prompt,
		// merging runtime and default options.
		Prompt requestPrompt = buildRequestPrompt(prompt);
		return internalStream(requestPrompt, null);
	}

	public Flux<ChatResponse> internalStream(Prompt prompt, ChatResponse previousChatResponse) {
		return Flux.deferContextual(contextView -> {
			ChatCompletionRequest request = createRequest(prompt, true);

			Flux<ChatCompletion> completionChunks = this.geminiApi.chatCompletionStream(request,
					getAdditionalHttpHeaders(prompt));

			final ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
				.prompt(prompt)
				.provider(org.springframework.ai.gemini.api.common.GeminiApiConstants.PROVIDER_NAME)
				.requestOptions(prompt.getOptions())
				.build();

			// Convert the ChatCompletionChunk into a ChatCompletion to be able to reuse
			// the function call handling logic.
			Flux<ChatResponse> chatResponse = completionChunks.map( chatCompletion -> {
				List<Generation> generations = chatCompletion.candidates().stream().map(candidate -> {
					return buildGeneration(candidate, Map.of(), request);
				})
				.toList();

				GeminiApi.Usage usage = chatCompletion.usage();
				Usage chatResponseUsage = usage != null ? getDefaultUsage(usage) : new EmptyUsage();
				return new ChatResponse(generations, from(chatCompletion, null, chatResponseUsage));
			});
////				.switchMap(generateContentResponse -> Mono.just(generateContentResponse).map(generateContentResponse2 -> {
////					try {
////						@SuppressWarnings("null")
//////						String id = generateContentResponse2.id();
////
////						List<Generation> generations = generateContentResponse2.candidates().stream().map(candidate -> { // @formatter:off
//////							if (candidate.content().role() != null) {
//////								roleMap.putIfAbsent(id, candidate.content().role().name());
//////							}
////							Map<String, Object> metadata = Map.of(
//////									"id", generateContentResponse2.id(),
//////									"role", roleMap.getOrDefault(id, "")
//////									,
//////									"index", candidate.index(),
//////									"finishReason", candidate.finishReason() != null ? candidate.finishReason().name() : ""
////							);
////
////							return buildGeneration(candidate, metadata, request);
////						}).toList();
////						// @formatter:on
////						GeminiApi.Usage usage = generateContentResponse2.usage();
////						Usage currentChatResponseUsage = usage != null ? getDefaultUsage(usage) : new EmptyUsage();
////						Usage accumulatedUsage = UsageUtils.getCumulativeUsage(currentChatResponseUsage,previousChatResponse);
////						return new ChatResponse(generations, from(generateContentResponse2, null, accumulatedUsage));
////					}
////					catch (Exception e) {
////						logger.error("Error processing chat completion", e);
////						return new ChatResponse(List.of());
////					}
////					// When in stream mode and enabled to include the usage, the Gemini
////					// Chat completion response would have the usage set only in its
////					// final response. Hence, the following overlapping buffer is
////					// created to store both the current and the subsequent response
////					// to accumulate the usage from the subsequent response.
////				}))
////				.buffer(2, 1)
////				.map(bufferList -> {
////					ChatResponse firstResponse = bufferList.get(0);
////					return firstResponse;
////				});
//
////			ChatResponse chatResponse = new ChatResponse(generations,
//					from(completionEntity.getBody(), rateLimit, accumulatedUsage));

			Flux<ChatResponse> flux = chatResponse.flatMap(Flux::just);

			return new MessageAggregator().aggregate(flux, observationContext::setResponse);

		});
	}

	private MultiValueMap<String, String> getAdditionalHttpHeaders(Prompt prompt) {

		Map<String, String> headers = new HashMap<>(this.defaultOptions.getHttpHeaders());
		if (prompt.getOptions() != null && prompt.getOptions() instanceof GeminiChatOptions chatOptions) {
			headers.putAll(chatOptions.getHttpHeaders());
		}
		return CollectionUtils.toMultiValueMap(
				headers.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, e -> List.of(e.getValue()))));
	}

	private Generation buildGeneration(Candidate candidate, Map<String, Object> metadata, ChatCompletionRequest request) {
		String finishReason = (candidate.finishReason() != null ? candidate.finishReason().name() : "");
		var generationMetadataBuilder = ChatGenerationMetadata.builder().finishReason(finishReason);

		List<Media> media = new ArrayList<>();
		String textContent = candidate.content() != null ? candidate.content().parts().get(0).text() : "";
		var assistantMessage = new AssistantMessage(textContent, metadata, List.of(), media);
		return new Generation(assistantMessage, generationMetadataBuilder.build());
	}

	private ChatResponseMetadata from(ChatCompletion result, RateLimit rateLimit, Usage usage) {
		Assert.notNull(result, "OpenAI ChatCompletionResult must not be null");
		var builder = ChatResponseMetadata.builder()
//				.id(result.id() != null ? result.id() : "")
				.usage(usage)
				.model(result.modelVersion() != null ? result.modelVersion() : "");
//				.keyValue("created", result.created() != null ? result.created() : 0L);
//				.keyValue("system-fingerprint", result.systemFingerprint() != null ? result.systemFingerprint() : "");
		if (rateLimit != null) {
			builder.rateLimit(rateLimit);
		}
		return builder.build();
	}


	private ChatResponseMetadata from(ChatResponseMetadata chatResponseMetadata, org.springframework.ai.chat.metadata.Usage usage) {
		Assert.notNull(chatResponseMetadata, "Gemini ChatResponseMetadata must not be null");
		var builder = ChatResponseMetadata.builder()
			.id(chatResponseMetadata.getId() != null ? chatResponseMetadata.getId() : "")
			.usage(usage)
			.model(chatResponseMetadata.getModel() != null ? chatResponseMetadata.getModel() : "");
		if (chatResponseMetadata.getRateLimit() != null) {
			builder.rateLimit(chatResponseMetadata.getRateLimit());
		}
		return builder.build();
	}

	private DefaultUsage getDefaultUsage(GeminiApi.Usage usage) {
		return new DefaultUsage(usage.promptTokenCount(), usage.candidatesTokenCount(), usage.totalTokenCount(), usage);
	}

	Prompt buildRequestPrompt(Prompt prompt) {
		// Process runtime options
		GeminiChatOptions runtimeOptions = null;
		if (prompt.getOptions() != null) {
			runtimeOptions = ModelOptionsUtils.copyToTarget(prompt.getOptions(), ChatOptions.class,
					GeminiChatOptions.class);
		}

		// Define request options by merging runtime options and default options
		GeminiChatOptions requestOptions = ModelOptionsUtils.merge(runtimeOptions, this.defaultOptions,
				GeminiChatOptions.class);

		// Merge @JsonIgnore-annotated options explicitly since they are ignored by
		// Jackson, used by ModelOptionsUtils.
		if (runtimeOptions != null) {
			requestOptions.setHttpHeaders(
					mergeHttpHeaders(runtimeOptions.getHttpHeaders(), this.defaultOptions.getHttpHeaders()));
		}
		else {
			requestOptions.setHttpHeaders(this.defaultOptions.getHttpHeaders());
		}

		return new Prompt(prompt.getInstructions(), requestOptions);
	}

	private Map<String, String> mergeHttpHeaders(Map<String, String> runtimeHttpHeaders,
			Map<String, String> defaultHttpHeaders) {
		var mergedHttpHeaders = new HashMap<>(defaultHttpHeaders);
		mergedHttpHeaders.putAll(runtimeHttpHeaders);
		return mergedHttpHeaders;
	}

	/**
	 * Accessible for testing.
	 */
	ChatCompletionRequest createRequest(Prompt prompt, boolean stream) {

		List<Content> contents = prompt.getInstructions().stream().map(message -> {
			if (message.getMessageType() == MessageType.USER) {
				String text = message.getText();
				return List.of(new Content(List.of(new Content.Part(text)),
						Content.Role.valueOf(message.getMessageType().name())));
			}
			else if (message.getMessageType() == MessageType.ASSISTANT || message.getMessageType() == MessageType.SYSTEM) {
				String text = message.getText();
				return List.of(new Content(List.of(new Content.Part(text)), Content.Role.MODEL));
			}
			else {
				throw new IllegalArgumentException("Unsupported content type: " + message.getMessageType());
			}
		}).flatMap(List::stream).toList();

		ChatCompletionRequest request = new ChatCompletionRequest(contents, stream);

		GeminiChatOptions requestOptions = (GeminiChatOptions) prompt.getOptions();
		request = ModelOptionsUtils.merge(requestOptions, request, ChatCompletionRequest.class);

		return request;
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return GeminiChatOptions.fromOptions(this.defaultOptions);
	}

	@Override
	public String toString() {
		return "GeminiChatModel [defaultOptions=" + this.defaultOptions + "]";
	}

	/**
	 * Use the provided convention for reporting observation data
	 * @param observationConvention The provided convention
	 */
	public void setObservationConvention(ChatModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private GeminiApi geminiApi;

		private GeminiChatOptions defaultOptions = GeminiChatOptions.builder()
			.model(GeminiApi.DEFAULT_CHAT_MODEL)
			.temperature(0.7)
			.build();

		private RetryTemplate retryTemplate = RetryUtils.DEFAULT_RETRY_TEMPLATE;

		private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

		private Builder() {
		}

		public Builder geminiApi(GeminiApi geminiApi) {
			this.geminiApi = geminiApi;
			return this;
		}

		public Builder defaultOptions(GeminiChatOptions defaultOptions) {
			this.defaultOptions = defaultOptions;
			return this;
		}

		public Builder retryTemplate(RetryTemplate retryTemplate) {
			this.retryTemplate = retryTemplate;
			return this;
		}

		public Builder observationRegistry(ObservationRegistry observationRegistry) {
			this.observationRegistry = observationRegistry;
			return this;
		}

		public GeminiChatModel build() {
			return new GeminiChatModel(geminiApi, defaultOptions, retryTemplate);
		}

	}

}
