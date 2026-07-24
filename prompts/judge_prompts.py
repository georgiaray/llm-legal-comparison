"""
NOTE: These are the exact evaluation/judge prompts used in the published study
this repository accompanies, to score the classification stage (climate
finance policy instruments) against a human-labeled ground truth set. They are
provided as a guide and source of inspiration for designing your own
LLM-as-judge evaluation prompts, not as a generic, plug-and-play template --
if you are adapting this framework for a different task, you will need to
write your own evaluation rubric tailored to your own criteria.

Two prompts are included:

1. The judge prompt (get_judge_prompt): scores a single classification
   response against ground truth on a continuous 0.0-1.0 scale, with a
   detailed rubric. An exact string match short-circuits this call and scores
   1.0 automatically, without invoking the judge.
2. The correction prompt (get_correction_prompt): a second-opinion prompt used
   in the optional review/correction step, which sees the full conversation
   that produced a response (system prompt, user prompt, assistant response)
   and returns structured feedback if a correction is needed.
"""

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for classification tasks. You assess whether "
    "responses are reasonable and demonstrate sound reasoning, not just exact matches."
)

CORRECTION_SYSTEM_PROMPT = (
    "You are an expert evaluator for climate finance classification tasks. Review "
    "responses critically using the full conversation context and provide "
    "constructive feedback."
)


def get_judge_prompt(question_description, groundtruth, model_response):
    """
    Build the judge prompt used to score a single classification response
    against ground truth. Returns a 0.0-1.0 score plus reasoning as JSON.

    Args:
        question_description: Short description of what's being classified
            (e.g. "Categorize whether the instrument primarily targets the
            financial sector or the real economy").
        groundtruth: The human-labeled ground truth answer for this question.
        model_response: The model's classification response being evaluated.
    """
    return f"""You are evaluating a classification task for climate finance policy instruments.

Question: {question_description}

Ground Truth Answer: {groundtruth}

Model Response: {model_response}

Your task is to assess whether the model response is reasonable and in line with the thinking of the ground truth evaluator. In general, be lenient if the reasoning is sound as this is all ambiguous.
This is not just about exact matching - consider:

1. **Reasonableness**: Is the model's response a reasonable interpretation of the policy instrument given the classification criteria?
2. **Alignment with thinking**: Does the model's response demonstrate similar reasoning and categorization logic as the ground truth?
3. **Correctness**: Does the response capture the same essential classification/categorization as the ground truth?

CRITICAL EVALUATION CRITERIA:

**Strict Adherence to Options**:
- The model response should only use categories, entities, market failures, and policy instruments that are in the classification schema
- If the response includes categories NOT in the provided list (e.g., "indigenous organizations", "non-profit organizations" when not listed), this is a significant issue
- However, note that "Public bodies" includes Indigenous organizations and governments

**Consistency of Terminology**:
- For Question 2, responses should use the EXACT category names from the schema consistently
- Do not accept responses that mix different phrasings for the same concept (e.g., alternating between "real economy mitigation" and "real economy decarbonization measures with a financial component")
- The ground truth uses specific terminology - the model should match that terminology, not use alternative phrasings

**Completeness**:
- For Question 5, responses must include BOTH the category AND the policy instrument (e.g., "Market-based: Concessional finance, subsidies and grants")
- If a response has the category but not the policy instrument, or vice versa, this is incomplete

**Question-Specific Considerations**:
- Question 2: Ensure consistent use of exact category names.
- Question 3: Verify all entities are from the provided list. The model should not add entities not in the schema
- Question 4: Consider that there are correlations - MIT measures often relate to externalities/public goods and concessional finance/subsidies/taxes. Market failures can be ambiguous, so be lenient if the reasoning is sound
- Question 5: Must include both category and policy instrument. Note that public investment in infrastructure counts as "Concessional finance, subsidies and grants" NOT "Market infrastructure"

Consider that:
- The response may use different wording but convey the same meaning (but prefer exact terminology matches)
- The response may be partially correct (e.g., identifies some but not all categories)
- Minor formatting differences are acceptable
- The core meaning, categories, and reasoning should align
- However, terminology consistency and strict adherence to provided options are important

Provide a nuanced score from 0.0 to 1.0 where:
- 1.0 = Perfect match or equivalent reasoning with correct terminology and complete response
- 0.8-0.9 = Very close, minor differences in wording or one missing element, but uses correct terminology
- 0.6-0.7 = Reasonable response that aligns with ground truth thinking but has some differences (terminology issues, missing elements, or going beyond options)
- 0.4-0.5 = Partially correct, shows some understanding but misses key elements or uses incorrect terminology
- 0.2-0.3 = Somewhat reasonable but significant differences from ground truth (wrong categories, incomplete, inconsistent)
- 0.0-0.1 = Incorrect or demonstrates different reasoning entirely

Format your response as JSON:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<detailed explanation addressing: (1) terminology consistency, (2) adherence to provided options, (3) completeness, (4) alignment with ground truth>"}}
"""


def get_correction_prompt(system_prompt, user_prompt, model_response):
    """
    Build the second-opinion/correction prompt, used to review a
    classification response with full access to the conversation that
    produced it (system prompt, user prompt, and the response itself).
    Returns either {} (no correction needed) or a structured feedback object.

    Args:
        system_prompt: The system prompt used for the original classification call.
        user_prompt: The full user prompt used for the original classification call
            (the original implementation truncates this to 2000 characters when
            building the correction prompt, to control context length).
        model_response: The assistant's response being reviewed.
    """
    truncated_user_prompt = (user_prompt or "")[:2000]
    return f"""You are an expert evaluator reviewing a classification response for a climate finance policy instrument.

You have access to the full conversation that led to this response. Here is the complete context:

SYSTEM PROMPT:
{system_prompt or 'N/A'}

USER PROMPT (includes document summary, relevant chunks, examples, and classification criteria):
{truncated_user_prompt or 'N/A'}...

ASSISTANT RESPONSE:
{model_response}

Review this response in the context of the full conversation. Consider:
- Does the response properly address all aspects of the user's prompt?
- Is the classification accurate given the document summary and relevant chunks provided?
- Does it follow the examples and classification criteria shown in the prompt?
- Are there any missing elements or incorrect categorizations?
- Could the response be improved in clarity, completeness, or accuracy?

Pay extra close attention to whether or not the specified output strucutre is followed. That should be penalized heavily if it is not followed.

If the response is correct, well-reasoned, and complete given the context, respond with:
{{}}

If corrections or improvements are needed, respond with:
{{
    "needs_correction": true,
    "feedback": "Specific, actionable feedback on what needs to be improved and how. Reference the context from the conversation (summary, chunks, examples) to explain what should be corrected."
}}

Format your response as JSON."""
