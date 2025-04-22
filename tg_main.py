import logging
import os
import asyncio
import time

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
)

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'project-nn'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_solution(question, max_length=512, temperature=0.7):
    prompt = f"{question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    solution = tokenizer.decode(output[0], skip_special_tokens=True)
    return solution.replace(prompt, "").strip()


# ------

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()

dp.include_router(router)

MODEL_BUTTONS = {
    'model_fast': 'Быстрая',
    'model_smart': 'Умная'
}

MODEL_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data=key)]
        for key, label in MODEL_BUTTONS.items()
    ]
)

CHANGE_MODEL_ONLY_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Сменить модель", callback_data="change_model")]
    ]
)

ACTION_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="Сгенерировать еще раз", callback_data="regenerate"),
            InlineKeyboardButton(text="Сменить модель", callback_data="change_model")
        ]
    ]
)

MODEL_MAP = {
    'model_fast': 'Быстрая (локальная)',
    'model_smart': 'Умная (локальная)'
}

user_models = {}
user_tasks = {}

@router.message(CommandStart())
async def cmd_start(message: Message):
    user_id = message.from_user.id
    user_models[user_id] = MODEL_MAP['model_fast']
    await message.answer(
        f'Здравствуйте, {message.chat.username}!\n'
        'Я бот, который помогает решать задачи по физике.\n'
        'Пожалуйста, выберите модель нейросети ниже или начните вводить вашу задачу.',
        reply_markup=MODEL_KEYBOARD,
        parse_mode='HTML'
    )

@router.callback_query(F.data.in_({'model_fast', 'model_smart'}))
async def process_model_selection(callback_query: CallbackQuery):
    selected_key = callback_query.data
    selected_model = MODEL_MAP.get(selected_key, MODEL_MAP['model_fast'])
    user_id = callback_query.from_user.id
    user_models[user_id] = selected_model

    label = MODEL_BUTTONS[selected_key]
    await callback_query.answer(f"Модель '{label}' выбрана.")
    await callback_query.message.edit_text(
        f"Модель успешно изменена на <b>{label}</b>.\n"
        "Теперь вы можете отправить вашу задачу по физике.",
        reply_markup=CHANGE_MODEL_ONLY_KEYBOARD,
        parse_mode='HTML'
    )

@router.callback_query(F.data == 'regenerate')
async def regenerate_response(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    if user_id not in user_tasks:
        await callback_query.answer("Пожалуйста, сначала отправьте вашу задачу.")
        return

    task = user_tasks[user_id]
    await callback_query.answer("Генерирую ответ заново...")

    start_time = time.time()
    answer = generate_solution(task)
    elapsed_time = round(time.time() - start_time, 2)

    await callback_query.message.answer(
        f"<b>Ответ от модели:</b>\n{answer}\n\n"
        f"<i>Время на ответ: {elapsed_time} секунд</i>",
        reply_markup=ACTION_KEYBOARD,
        parse_mode='HTML'
    )

@router.callback_query(F.data == 'change_model')
async def change_model(callback_query: CallbackQuery):
    await callback_query.answer("Пожалуйста, выберите модель нейросети.")
    await callback_query.message.edit_text(
        "Выберите модель нейросети:",
        reply_markup=MODEL_KEYBOARD,
        parse_mode='HTML'
    )

@router.message(F.text)
async def answering(message: Message):
    user_id = message.from_user.id
    task = message.text.strip()
    selected_model = user_models.get(user_id, MODEL_MAP['model_fast'])

    start_time = time.time()
    answer = generate_solution(task)
    elapsed_time = round(time.time() - start_time, 2)

    await message.answer(
        f"<b>Ответ от модели:</b>\n{answer}\n\n"
        f"<i>Время на ответ: {elapsed_time} секунд</i>",
        reply_markup=ACTION_KEYBOARD,
        parse_mode='HTML'
    )
    user_tasks[user_id] = task


async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Выход по Ctrl+C')
