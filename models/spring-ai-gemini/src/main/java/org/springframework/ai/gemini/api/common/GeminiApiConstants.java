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

package org.springframework.ai.gemini.api.common;

import org.springframework.ai.observation.conventions.AiProvider;

/**
 * Common value constants for Gemini api.
 *
 * @author Jonghoon Park
 */
public final class GeminiApiConstants {

	public static final String DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com";

	// todo: replace
//	public static final String PROVIDER_NAME = AiProvider.GEMINI.value();
	public static final String PROVIDER_NAME = "GEMINI";

	private GeminiApiConstants() {

	}

}
