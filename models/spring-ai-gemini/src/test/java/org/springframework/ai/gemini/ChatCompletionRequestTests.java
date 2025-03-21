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

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.gemini.api.GeminiApi;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Jonghoon Park
 */
class ChatCompletionRequestTests {

	@Test
	void createRequestWithChatOptions() {
		var client = GeminiChatModel.builder().geminiApi(GeminiApi.builder().apiKey("TEST").build())
						.defaultOptions(GeminiChatOptions.builder().model("DEFAULT_MODEL").temperature(66.6).build())
						.build();

		var prompt = client.buildRequestPrompt(new Prompt("Test content content"));

		var request = client.createRequest(prompt, false);

		assertThat(request.contents()).hasSize(1);
		assertThat(request.stream()).isFalse();

		assertThat(request.model()).isEqualTo("DEFAULT_MODEL");
		assertThat(request.temperature()).isEqualTo(66.6);

		request = client.createRequest(new Prompt("Test content content",
				org.springframework.ai.gemini.GeminiChatOptions.builder().model("PROMPT_MODEL").temperature(99.9).build()), true);

		assertThat(request.contents()).hasSize(1);
		assertThat(request.stream()).isTrue();

		assertThat(request.model()).isEqualTo("PROMPT_MODEL");
		assertThat(request.temperature()).isEqualTo(99.9);
	}

}
