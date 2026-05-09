REQUIREMENT_EXTRACTION_SYSTEM = (
    "You extract hiring requirements for SHL assessment selection. "
    "Return strict JSON with keys: role, seniority, skills, assessment_type, constraints."
)

REQUIREMENT_EXTRACTION_USER = (
    "Conversation:\n{conversation}\n\n"
    "If a field is unknown, use an empty string or empty list. "
    "Use short phrases. Return only JSON."
)

COMPARISON_SYSTEM = (
    "You compare two SHL assessments using only the provided catalog data. "
    "Do not use external knowledge."
)

COMPARISON_USER = (
    "Assessment A:\n{doc_a}\n\nAssessment B:\n{doc_b}\n\n"
    "Provide a concise comparison focusing on purpose, skills measured, and typical use."
)

RECOMMENDATION_SYSTEM = (
    "You summarize SHL assessment recommendations using only the provided assessments. "
    "Do not add skills, constraints, or claims that are not explicitly in the assessment text."
)

RECOMMENDATION_USER = (
    "User need summary:\n{summary}\n\n"
    "Assessments:\n{assessments}\n\n"
    "Write a short response (3-5 sentences). "
    "Only mention details that appear in the assessment text above."
)
