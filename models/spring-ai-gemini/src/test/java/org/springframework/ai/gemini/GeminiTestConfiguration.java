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

import org.springframework.ai.model.ApiKey;
import org.springframework.ai.model.SimpleApiKey;
import org.springframework.ai.gemini.api.GeminiApi;
import org.springframework.ai.gemini.api.GeminiApi.ChatModel;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

@SpringBootConfiguration
public class GeminiTestConfiguration {

	@Bean
	public GeminiApi openAiApi() {
		return GeminiApi.builder().apiKey(getApiKey()).build();
	}

	private ApiKey getApiKey() {
		String apiKey = System.getenv("GEMINI_API_KEY");
		if (!StringUtils.hasText(apiKey)) {
			throw new IllegalArgumentException(
					"You must provide an API key.  Put it in an environment variable under the name GEMINI_API_KEY");
		}
		return new SimpleApiKey(apiKey);
	}

	@Bean
	public GeminiChatModel openAiChatModel(GeminiApi api) {
		return GeminiChatModel.builder()
				.geminiApi(api)
				.defaultOptions(GeminiChatOptions.builder().model(ChatModel.GEMINI_2_O_FLASH).build())
				.build();
	}

	@Bean
	public GeminiEmbeddingModel openAiEmbeddingModel(GeminiApi api) {
		return new GeminiEmbeddingModel(api);
	}

}
