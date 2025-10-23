from pydantic import BaseModel, Field

class SingleSampleScoringResponse(BaseModel):
    score: int = Field(
        ..., description="1 if the sample matches the feature description as expected, 0 otherwise."
    )
    explanation: str = Field(
        ..., description="Explanation for the score."
    )


class FeatureLabelResponse(BaseModel):
    label: str = Field(
        ..., description="A concise phrase describing the property present in the positive samples (considering both context and marked tokens) but not in the negative samples."
    )
    brief_description: str = Field(
        ..., description="A sentence expanding on the label, explaining what the feature is detecting in more detail."
    )
    detailed_explanation: str | None = Field(
        ..., description="An extended explanation of what this feature is detecting, including how the context before the marked tokens contributes to the feature's meaning."
    )