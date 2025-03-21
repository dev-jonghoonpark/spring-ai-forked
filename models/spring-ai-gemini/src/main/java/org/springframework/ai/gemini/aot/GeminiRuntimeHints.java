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

package org.springframework.ai.gemini.aot;

import java.util.Set;

import org.springframework.ai.gemini.api.GeminiApi;
import org.springframework.aot.hint.MemberCategory;
import org.springframework.aot.hint.RuntimeHints;
import org.springframework.aot.hint.RuntimeHintsRegistrar;
import org.springframework.aot.hint.TypeReference;
import org.springframework.lang.NonNull;
import org.springframework.lang.Nullable;

import static org.springframework.ai.aot.AiRuntimeHints.findJsonAnnotatedClassesInPackage;

/**
 * The GeminiRuntimeHints class is responsible for registering runtime hints for Gemini
 * API classes.
 *
 * @author Jonghoon Park
 */
public class GeminiRuntimeHints implements RuntimeHintsRegistrar {

	private static Set<TypeReference> eval(Set<TypeReference> referenceSet) {
		referenceSet.forEach(tr -> System.out.println(tr.toString()));
		return referenceSet;
	}

	@Override
	public void registerHints(@NonNull RuntimeHints hints, @Nullable ClassLoader classLoader) {
		var mcs = MemberCategory.values();
		for (var tr : eval(findJsonAnnotatedClassesInPackage(GeminiApi.class))) {
			hints.reflection().registerType(tr, mcs);
		}
	}

}
