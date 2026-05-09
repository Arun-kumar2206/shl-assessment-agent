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
    "You are an assistant that summarizes SHL assessment recommendations. "
    "Only refer to the provided assessments."
)

RECOMMENDATION_USER = (
    "User need summary:\n{summary}\n\n"
    "Assessments:\n{assessments}\n\n"
    "Write a short response that introduces the shortlist and highlights fit."
)
