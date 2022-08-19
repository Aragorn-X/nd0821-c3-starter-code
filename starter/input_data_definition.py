from pydantic import BaseModel, Field
from typing import Union



class InputStructure(BaseModel):
    age: int
    workclass: Union[str, list]
    fnlgt: int
    education: Union[str, list]
    education_num: int = Field(alias="education-num")
    marital_status: Union[str, list] = Field(alias="marital-status")
    occupation: Union[str, list]
    relationship: Union[str, list]
    race: Union[str, list]
    sex: Union[str, list]
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: Union[str, list] = Field(alias="native-country")
    salary: int

    class Config:
        allow_population_by_field_name = True


