
import os
import asyncio
import base64

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")


class PddOutputSchema(BaseModel):
    class_id: int
    class_name: str
    category: str
    description: str


parser = PydanticOutputParser(pydantic_object=PddOutputSchema)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты специалист по правилам дорожного движения. "
            "Твоя задача — распознать дорожный знак на фотографии. "
            "class_id — номер класса знака в датасете GTSRB, "
            "целое число от 0 до 42. Например, 14 — знак Stop. "
            "class_name — название знака на английском языке, "
            "например Stop. "
            "category — категория знака на русском языке: "
            "Запрещающий знак, Предупреждающий знак, "
            "Предписывающий знак, Знак приоритета или Знак отмены. "
            "description — краткое описание знака на русском языке, "
            "одно предложение. Например: Остановка обязательна. "
            "Если дорожного знака на фотографии нет, "
            "верни class_id равный -1, а в class_name, category "
            "и description укажи прочерк. "
            "Не выдумывай информацию, которой нет на изображении. "
            "{format_instructions}",
        ),
        (
            "human",
            [
                {
                    "type": "text",
                    "text": "Какой дорожный знак изображён на этой фотографии?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:{mime};base64,{image}"
                    },
                },
            ],
        ),
    ]
).partial(format_instructions=parser.get_format_instructions())

model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

chain = prompt | model | parser

router = APIRouter(prefix="/pdd", tags=["Распознавание дорожных знаков"])


@router.post(
    path="/predict/",
    response_model=PddOutputSchema
)
async def check_image_pdd(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail="Неверный формат файла. Загрузите изображение."
        )

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=422,
            detail="Загруженный файл пуст."
        )

    encoded_image = base64.b64encode(raw).decode("utf-8")

    try:
        result = await asyncio.wait_for(
            chain.ainvoke({
                "mime": file.content_type,
                "image": encoded_image
            }),
            timeout=120
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=500,
            detail="Модель не успела выполнить задачу за отведённое время."
        )

    except OutputParserException:
        raise HTTPException(
            status_code=500,
            detail="Ответ модели не соответствует заданной схеме."
        )

    if result.class_id == -1:
        raise HTTPException(
            status_code=422,
            detail="На изображении не обнаружен дорожный знак."
        )

    return result


app = FastAPI(title="Распознавание дорожных знаков")

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
