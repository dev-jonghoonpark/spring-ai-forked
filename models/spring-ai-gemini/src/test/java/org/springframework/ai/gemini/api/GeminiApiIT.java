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

package org.springframework.ai.gemini.api;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import reactor.core.publisher.Flux;

import org.springframework.ai.gemini.api.GeminiApi.ChatCompletion;
import org.springframework.ai.gemini.api.GeminiApi.ChatCompletionChunk;
import org.springframework.ai.gemini.api.GeminiApi.Content;
import org.springframework.ai.gemini.api.GeminiApi.Content.Role;
import org.springframework.ai.gemini.api.GeminiApi.ChatCompletionRequest;
import org.springframework.ai.gemini.api.GeminiApi.Embedding;
import org.springframework.ai.gemini.api.GeminiApi.EmbeddingList;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * @author Jonghoon Park
 */
@EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".+")
public class GeminiApiIT {

	GeminiApi geminiApi = GeminiApi.builder().apiKey(System.getenv("GEMINI_API_KEY")).build();

	@Test
	void chatCompletionEntity() {
		Content content = new Content(List.of(new Content.Part("Hello world")), Role.USER);
		ResponseEntity<ChatCompletion> response = this.geminiApi.chatCompletionEntity(
				new ChatCompletionRequest(List.of(content), GeminiApi.DEFAULT_CHAT_MODEL.getValue(), 0.8, false));

		assertThat(response).isNotNull();
		assertThat(response.getBody()).isNotNull();
	}

	@Test
	void chatCompletionStream() {
		Content content = new Content(List.of(new Content.Part("Hello world")), Role.USER);
		Flux<ChatCompletion> response = this.geminiApi.chatCompletionStream(
				new ChatCompletionRequest(List.of(content), GeminiApi.DEFAULT_CHAT_MODEL.getValue(), 0.8, true));

		assertThat(response).isNotNull();
		assertThat(response.collectList().block()).isNotNull();
	}

	@Test
	void embeddings() {
		ResponseEntity<EmbeddingList<Embedding>> response = this.geminiApi
			.embeddings(new GeminiApi.EmbeddingRequest<String>("Hello world"));

		assertThat(response).isNotNull();
		assertThat(response.getBody().data()).hasSize(1);
		assertThat(response.getBody().data().get(0).embedding()).hasSize(1536);
	}

}
