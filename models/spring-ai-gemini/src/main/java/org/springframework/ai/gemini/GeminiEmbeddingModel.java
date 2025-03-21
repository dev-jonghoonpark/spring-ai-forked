/*
 * Copyright 2023-2024 the original author or authors.
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

import java.util.List;

import io.micrometer.observation.ObservationRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.embedding.AbstractEmbeddingModel;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.EmbeddingResponseMetadata;
import org.springframework.ai.embedding.observation.DefaultEmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationContext;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationDocumentation;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.gemini.api.GeminiApi;
import org.springframework.ai.gemini.api.GeminiApi.Usage;
import org.springframework.ai.gemini.api.GeminiApi.EmbeddingList;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.lang.Nullable;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;

/**
 * Open AI Embedding Model implementation.
 *
 * @author Jonghoon Park
 */
public class GeminiEmbeddingModel extends AbstractEmbeddingModel {

	private static final Logger logger = LoggerFactory.getLogger(GeminiEmbeddingModel.class);

	private static final EmbeddingModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultEmbeddingModelObservationConvention();

	private final org.springframework.ai.gemini.GeminiEmbeddingOptions defaultOptions;

	private final RetryTemplate retryTemplate;

	private final GeminiApi geminiApi;

	private final MetadataMode metadataMode;

	/**
	 * Observation registry used for instrumentation.
	 */
	private final ObservationRegistry observationRegistry;

	/**
	 * Conventions to use for generating observations.
	 */
	private EmbeddingModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	/**
	 * Constructor for the GeminiEmbeddingModel class.
	 * @param geminiApi The GeminiApi instance to use for making API requests.
	 */
	public GeminiEmbeddingModel(GeminiApi geminiApi) {
		this(geminiApi, MetadataMode.EMBED);
	}

	/**
	 * Initializes a new instance of the GeminiEmbeddingModel class.
	 * @param geminiApi The GeminiApi instance to use for making API requests.
	 * @param metadataMode The mode for generating metadata.
	 */
	public GeminiEmbeddingModel(GeminiApi geminiApi, MetadataMode metadataMode) {
		this(geminiApi, metadataMode,
				org.springframework.ai.gemini.GeminiEmbeddingOptions.builder().model(GeminiApi.DEFAULT_EMBEDDING_MODEL).build());
	}

	/**
	 * Initializes a new instance of the GeminiEmbeddingModel class.
	 * @param geminiApi The GeminiApi instance to use for making API requests.
	 * @param metadataMode The mode for generating metadata.
	 * @param geminiEmbeddingOptions The options for Gemini embedding.
	 */
	public GeminiEmbeddingModel(GeminiApi geminiApi, MetadataMode metadataMode,
								org.springframework.ai.gemini.GeminiEmbeddingOptions geminiEmbeddingOptions) {
		this(geminiApi, metadataMode, geminiEmbeddingOptions, RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	/**
	 * Initializes a new instance of the GeminiEmbeddingModel class.
	 * @param geminiApi - The GeminiApi instance to use for making API requests.
	 * @param metadataMode - The mode for generating metadata.
	 * @param options - The options for Gemini embedding.
	 * @param retryTemplate - The RetryTemplate for retrying failed API requests.
	 */
	public GeminiEmbeddingModel(GeminiApi geminiApi, MetadataMode metadataMode, org.springframework.ai.gemini.GeminiEmbeddingOptions options,
								RetryTemplate retryTemplate) {
		this(geminiApi, metadataMode, options, retryTemplate, ObservationRegistry.NOOP);
	}

	/**
	 * Initializes a new instance of the GeminiEmbeddingModel class.
	 * @param geminiApi - The GeminiApi instance to use for making API requests.
	 * @param metadataMode - The mode for generating metadata.
	 * @param options - The options for Gemini embedding.
	 * @param retryTemplate - The RetryTemplate for retrying failed API requests.
	 * @param observationRegistry - The ObservationRegistry used for instrumentation.
	 */
	public GeminiEmbeddingModel(GeminiApi geminiApi, MetadataMode metadataMode, org.springframework.ai.gemini.GeminiEmbeddingOptions options,
								RetryTemplate retryTemplate, ObservationRegistry observationRegistry) {
		Assert.notNull(geminiApi, "geminiApi must not be null");
		Assert.notNull(metadataMode, "metadataMode must not be null");
		Assert.notNull(options, "options must not be null");
		Assert.notNull(retryTemplate, "retryTemplate must not be null");
		Assert.notNull(observationRegistry, "observationRegistry must not be null");

		this.geminiApi = geminiApi;
		this.metadataMode = metadataMode;
		this.defaultOptions = options;
		this.retryTemplate = retryTemplate;
		this.observationRegistry = observationRegistry;
	}

	@Override
	public float[] embed(Document document) {
		Assert.notNull(document, "Document must not be null");
		return this.embed(document.getFormattedContent(this.metadataMode));
	}

	@Override
	public EmbeddingResponse call(EmbeddingRequest request) {
		org.springframework.ai.gemini.GeminiEmbeddingOptions requestOptions = mergeOptions(request.getOptions(), this.defaultOptions);
		GeminiApi.EmbeddingRequest<List<String>> apiRequest = createRequest(request, requestOptions);

		var observationContext = EmbeddingModelObservationContext.builder()
			.embeddingRequest(request)
			.provider(org.springframework.ai.gemini.api.common.GeminiApiConstants.PROVIDER_NAME)
			.requestOptions(requestOptions)
			.build();

		return EmbeddingModelObservationDocumentation.EMBEDDING_MODEL_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {
				EmbeddingList<GeminiApi.Embedding> apiEmbeddingResponse = this.retryTemplate
					.execute(ctx -> this.geminiApi.embeddings(apiRequest).getBody());

				if (apiEmbeddingResponse == null) {
					logger.warn("No embeddings returned for request: {}", request);
					return new EmbeddingResponse(List.of());
				}

				var metadata = new EmbeddingResponseMetadata(apiEmbeddingResponse.model(),
						getDefaultUsage(apiEmbeddingResponse.usage()));

				List<Embedding> embeddings = apiEmbeddingResponse.data()
					.stream()
					.map(e -> new Embedding(e.embedding(), e.index()))
					.toList();

				EmbeddingResponse embeddingResponse = new EmbeddingResponse(embeddings, metadata);

				observationContext.setResponse(embeddingResponse);

				return embeddingResponse;
			});
	}

	private DefaultUsage getDefaultUsage(Usage usage) {
		return new DefaultUsage(usage.promptTokenCount(), usage.candidatesTokenCount(), usage.totalTokenCount(), usage);
	}

	private GeminiApi.EmbeddingRequest<List<String>> createRequest(EmbeddingRequest request,
                                                                   org.springframework.ai.gemini.GeminiEmbeddingOptions requestOptions) {
		return new GeminiApi.EmbeddingRequest<>(request.getInstructions(), requestOptions.getModel(),
				requestOptions.getEncodingFormat(), requestOptions.getDimensions(), requestOptions.getUser());
	}

	/**
	 * Merge runtime and default {@link EmbeddingOptions} to compute the final options to
	 * use in the request.
	 */
	private org.springframework.ai.gemini.GeminiEmbeddingOptions mergeOptions(@Nullable EmbeddingOptions runtimeOptions,
																			  org.springframework.ai.gemini.GeminiEmbeddingOptions defaultOptions) {
		var runtimeOptionsForProvider = ModelOptionsUtils.copyToTarget(runtimeOptions, EmbeddingOptions.class,
				org.springframework.ai.gemini.GeminiEmbeddingOptions.class);

		if (runtimeOptionsForProvider == null) {
			return defaultOptions;
		}

		return org.springframework.ai.gemini.GeminiEmbeddingOptions.builder()
			// Handle portable embedding options
			.model(ModelOptionsUtils.mergeOption(runtimeOptionsForProvider.getModel(), defaultOptions.getModel()))
			.dimensions(ModelOptionsUtils.mergeOption(runtimeOptionsForProvider.getDimensions(),
					defaultOptions.getDimensions()))
			// Handle Gemini specific embedding options
			.encodingFormat(ModelOptionsUtils.mergeOption(runtimeOptionsForProvider.getEncodingFormat(),
					defaultOptions.getEncodingFormat()))
			.user(ModelOptionsUtils.mergeOption(runtimeOptionsForProvider.getUser(), defaultOptions.getUser()))
			.build();
	}

	/**
	 * Use the provided convention for reporting observation data
	 * @param observationConvention The provided convention
	 */
	public void setObservationConvention(EmbeddingModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

}
