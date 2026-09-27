"""Prompt engineering module for transcript polishing and multi-level summarization.

All prompts are provider-agnostic and adhere to strict editorial guidelines.
"""


def get_polish_prompt(transcript: str) -> tuple[str, str]:
    """Generate prompt to polish transcripts.

    Removes verbal disfluencies, stuttering, and filler words,
    fixes punctuation and speech-to-text phonetic errors,
    while strictly preserving the speaker's original meaning and words.
    """
    system_prompt = (
        "You are an expert audio transcript editor and verbatim proofreader. "
        "Your role is to polish raw speech-to-text transcripts into clean, fluent, readable text.\n"
        "Guidelines:\n"
        "1. Remove verbal disfluencies, stammering, and filler sounds (e.g., 'um', 'uh', 'er', 'ah', 'like, you know').\n"
        "2. Fix punctuation, capitalization, and paragraph spacing for clarity.\n"
        "3. Correct evident speech-to-text phonetic misrecognitions while preserving technical terminology.\n"
        "4. DO NOT alter the speaker's meaning, tone, facts, or key vocabulary.\n"
        "5. If speaker labels (e.g., '[Speaker 0]:' or '[Alice]:') are present, strictly preserve speaker attribution headers.\n"
        "6. Output ONLY the polished transcript. Do not prepend greetings, explanations, or commentary."
    )
    user_prompt = f"Please polish the following audio transcript:\n\n{transcript.strip()}"
    return system_prompt, user_prompt


def get_summary_prompt(
    transcript: str,
    level: str = "bullets",
    custom_instruction: str | None = None
) -> tuple[str, str]:
    """Generate prompt for multi-level summarization.

    Supported detail levels:
    - 'tldr': 1-2 sentence executive snapshot.
    - 'bullets': Key discussion takeaways and points.
    - 'detailed': Comprehensive structured executive breakdown.
    - 'action_items': Extracted action items, deliverables, and owners.
    - 'custom': User-defined instruction.
    """
    norm_level = level.lower().strip()

    if norm_level == "tldr":
        system_prompt = (
            "You are an executive assistant. Produce an ultra-concise, 1 to 2 sentence TL;DR snapshot "
            "capturing the core essence and conclusion of the transcript. Output ONLY the TL;DR sentence."
        )
        user_prompt = f"Transcript:\n\n{transcript.strip()}"

    elif norm_level == "bullets":
        system_prompt = (
            "You are an executive summary specialist. Summarize the transcript into clear, "
            "high-impact bullet points highlighting key insights, decisions, and takeaways. "
            "If speaker labels are present, attribute key viewpoints or commitments to the respective speakers. "
            "Organize them logically and avoid unnecessary filler."
        )
        user_prompt = f"Please extract the key takeaways in bullet points from this transcript:\n\n{transcript.strip()}"

    elif norm_level == "detailed":
        system_prompt = (
            "You are a professional documentation specialist. Provide a detailed, structured summary "
            "formatted in Markdown with the following sections:\n"
            "- ## Overview\n"
            "- ## Key Discussion Topics & Decisions\n"
            "- ## Detailed Insights\n"
            "- ## Conclusion\n"
            "If speaker labels are present, attribute discussion points and decisions to the respective participants. "
            "Be thorough, objective, and accurately reflect all nuances."
        )
        user_prompt = f"Provide a comprehensive, detailed summary of this transcript:\n\n{transcript.strip()}"

    elif norm_level == "action_items":
        system_prompt = (
            "You are an agile project manager. Analyze the transcript and extract all action items, "
            "tasks, commitments, and deadlines.\n"
            "Format as a Markdown checklist (- [ ] Task description [Assignee/Speaker]). "
            "If speaker labels are present, attribute tasks directly to the identified speaker. "
            "If no action items are present, explicitly state 'No actionable tasks identified.'"
        )
        user_prompt = f"Extract all action items and next steps from this transcript:\n\n{transcript.strip()}"

    elif norm_level == "custom" and custom_instruction:
        system_prompt = (
            "You are an expert AI assistant. Analyze the provided audio transcript according to the user's specific instruction."
        )
        user_prompt = f"Instruction: {custom_instruction}\n\nTranscript:\n\n{transcript.strip()}"

    else:
        system_prompt = "Summarize the key points of the transcript clearly and concisely."
        user_prompt = f"Transcript:\n\n{transcript.strip()}"

    return system_prompt, user_prompt
