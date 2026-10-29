# Program by Kaliyev.A
import telebot
import numpy as np
import joblib
from telebot import types
import logging
import time
import os
import openai
import pandas as pd

# Загрузка модели из файла
model = joblib.load('model.pkl')

# Укажите свой ключ API OpenAI
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    ""
)

MODEL = "gpt-4o-mini"
user_data = {}

# Токен телеграм-бота
API_TOKEN = ''

bot = telebot.TeleBot(API_TOKEN)
openai.api_key = OPENAI_API_KEY

disclaimer = (
    "⚠️ Внимание! Этот бот предназначен для образовательных целей и "
    "не заменяет консультацию с медицинским специалистом. Если у вас есть подозрения на проблемы со здоровьем, "
    "обратитесь к врачу."
)

def remove_keyboard():
    return types.ReplyKeyboardRemove()

def create_gender_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Мужской", "Женский")
    return keyboard

def create_cp_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Отсутствие боли", "Небольшая боль", "Умеренная боль", "Сильная боль")
    return keyboard

def create_fbs_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Да", "Нет")
    return keyboard

def create_ecg_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Норма", "Отклонение", "Гипертрофия")
    return keyboard

def create_exang_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Да", "Нет")
    return keyboard

def create_slope_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Пониженный", "Нормальный", "Повышенный")
    return keyboard

def create_thal_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Норм", "Дефект", "Обратимый дефект")
    return keyboard

def create_recommendation_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Дать рекомендации", "Пропустить")
    return keyboard

def create_after_recommendation_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Задать вопрос по рекомендациям", "Завершить")
    return keyboard

def create_gpt_chat_keyboard():
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    keyboard.add("Завершить диалог")
    return keyboard

@bot.message_handler(commands=['break'])
def break_process(message):
    chat_id = message.chat.id
    if chat_id in user_data:
        del user_data[chat_id]
    bot.send_message(chat_id, "Диагностика прервана. Если хотите начать заново, введите /start")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, disclaimer)
    bot.send_message(message.chat.id,
                     "Вы можете в любой момент прекратить диагностику нажав /break\nПожалуйста, введите ваш возраст (в годах):")
    user_data[message.chat.id] = {'state': 'ask_age'}

@bot.message_handler(func=lambda m: True)
def handle_user_input(m):
    chat_id = m.chat.id
    text = m.text.strip().lower()

    if chat_id not in user_data:
        bot.send_message(chat_id, "Спасибо за использование! Для повторного анализа начните с /start")
        return

    state = user_data[chat_id].get('state', None)

    # Маппинги для выбора значений
    cp_map = {
        "отсутствие боли": 0,
        "небольшая боль": 1,
        "умеренная боль": 2,
        "сильная боль": 3
    }
    ecg_map = {"норма": 0, "отклонение": 1, "гипертрофия": 2}
    slope_map = {"пониженный": 0, "нормальный": 1, "повышенный": 2}
    thal_map = {"норм": 0, "дефект": 1, "обратимый дефект": 2}

    if state == 'ask_age':
        try:
            age = int(m.text)
            if age < 0 or age > 120:
                raise ValueError
            user_data[chat_id]['age'] = age
            user_data[chat_id]['state'] = 'ask_sex'
            bot.send_message(chat_id, "Выберите ваш пол:", reply_markup=create_gender_keyboard())
        except:
            bot.send_message(chat_id, "Пожалуйста, введите корректный возраст.")

    elif state == 'ask_sex':
        if text == "мужской":
            user_data[chat_id]['sex'] = 1
        elif text == "женский":
            user_data[chat_id]['sex'] = 0
        else:
            bot.send_message(chat_id, "Выберите пол с помощью кнопок.")
            return
        user_data[chat_id]['state'] = 'ask_cp'
        bot.send_message(chat_id, "Выберите тип боли в груди:", reply_markup=create_cp_keyboard())

    elif state == 'ask_cp':
        cp = cp_map.get(m.text.lower())
        if cp is None:
            bot.send_message(chat_id, "Выберите тип боли в груди с помощью кнопок.")
            return
        user_data[chat_id]['cp'] = cp
        user_data[chat_id]['state'] = 'ask_trestbps'
        bot.send_message(chat_id, "Введите уровень систолического давления (мм рт. ст.):\nПример:120", reply_markup=remove_keyboard())

    elif state == 'ask_trestbps':
        try:
            val = float(m.text)
            if val <= 0:
                raise ValueError
            user_data[chat_id]['trestbps'] = val
            user_data[chat_id]['state'] = 'ask_chol'
            bot.send_message(chat_id, "Введите уровень холестерина (мг/дл):\nПример:200")
        except:
            bot.send_message(chat_id, "Введите корректное число для давления.")

    elif state == 'ask_chol':
        try:
            val = float(m.text)
            if val <= 0:
                raise ValueError
            user_data[chat_id]['chol'] = val
            user_data[chat_id]['state'] = 'ask_fbs'
            bot.send_message(chat_id, "Уровень сахара в крови натощак > 120 мг/дл?", reply_markup=create_fbs_keyboard())
        except:
            bot.send_message(chat_id, "Введите корректное число для холестерина.")

    elif state == 'ask_fbs':
        if m.text.lower() == "да":
            user_data[chat_id]['fbs'] = 1
        elif m.text.lower() == "нет":
            user_data[chat_id]['fbs'] = 0
        else:
            bot.send_message(chat_id, "Выберите уровень сахара с помощью кнопок.")
            return
        user_data[chat_id]['state'] = 'ask_restecg'
        bot.send_message(chat_id, "Выберите результат ЭКГ:", reply_markup=create_ecg_keyboard())

    elif state == 'ask_restecg':
        val = ecg_map.get(m.text.lower())
        if val is None:
            bot.send_message(chat_id, "Выберите результат ЭКГ с помощью кнопок.")
            return
        user_data[chat_id]['restecg'] = val
        user_data[chat_id]['state'] = 'ask_thalach'
        bot.send_message(chat_id, "Введите максимальную ЧСС:\nПример:110", reply_markup=remove_keyboard())

    elif state == 'ask_thalach':
        try:
            val = float(m.text)
            if val <= 0:
                raise ValueError
            user_data[chat_id]['thalach'] = val
            user_data[chat_id]['state'] = 'ask_exang'
            bot.send_message(chat_id, "Была ли стенокардия при нагрузке?", reply_markup=create_exang_keyboard())
        except:
            bot.send_message(chat_id, "Введите корректное число для ЧСС.")

    elif state == 'ask_exang':
        if m.text.lower() == "да":
            user_data[chat_id]['exang'] = 1
        elif m.text.lower() == "нет":
            user_data[chat_id]['exang'] = 0
        else:
            bot.send_message(chat_id, "Выберите с помощью кнопок.")
            return
        user_data[chat_id]['state'] = 'ask_oldpeak'
        bot.send_message(chat_id, "Введите уровень депрессии ST:\nПример:1.5", reply_markup=remove_keyboard())

    elif state == 'ask_oldpeak':
        try:
            val = float(m.text)
            if val < 0:
                raise ValueError
            user_data[chat_id]['oldpeak'] = val
            user_data[chat_id]['state'] = 'ask_slope'
            bot.send_message(chat_id, "Введите наклон сегмента ST:", reply_markup=create_slope_keyboard())
        except:
            bot.send_message(chat_id, "Введите корректное число для oldpeak.")

    elif state == 'ask_slope':
        val = slope_map.get(m.text.lower())
        if val is None:
            bot.send_message(chat_id, "Выберите наклон ST с помощью кнопок.")
            return
        user_data[chat_id]['slope'] = val
        user_data[chat_id]['state'] = 'ask_ca'
        bot.send_message(chat_id, "Введите количество крупных сосудов (0-4), окрашенных при флюороскопии:")

    elif state == 'ask_ca':
        try:
            ca = int(m.text)
            if ca < 0 or ca > 4:
                raise ValueError
            user_data[chat_id]['ca'] = ca
            user_data[chat_id]['state'] = 'ask_thal'
            bot.send_message(chat_id, "Введите результат теста на талассемию:", reply_markup=create_thal_keyboard())
        except:
            bot.send_message(chat_id, "Введите число 0-4.")

    elif state == 'ask_thal':
        val = thal_map.get(m.text.lower())
        if val is None:
            bot.send_message(chat_id, "Выберите талассемию с помощью кнопок.")
            return
        user_data[chat_id]['thal'] = val
        send_prediction(chat_id)

    elif state == 'after_prediction':
        if text == "дать рекомендации":
            recommendations = get_recommendations(chat_id)
            bot.send_message(chat_id, recommendations)
            # Инициализируем историю диалога с GPT с базовым контекстом
            user_data[chat_id]['gpt_history'] = [
                {"role": "system", "content": "Ты опытный кардиолог с 20-летним стажем. Отвечай кратко и понятно, без маркированных списков."},
                {"role": "assistant", "content": recommendations}
            ]
            bot.send_message(chat_id,
                             "Диалог с GPT открыт. Задавайте свои вопросы по рекомендациям. Для завершения нажмите кнопку 'Завершить диалог'.",
                             reply_markup=create_gpt_chat_keyboard())
            user_data[chat_id]['state'] = 'gpt_chat'
        elif text == "пропустить":
            bot.send_message(chat_id,
                             "Завершено. Если хотите начать заново, введите /start",
                             reply_markup=remove_keyboard())
            del user_data[chat_id]
        else:
            bot.send_message(chat_id, "Выберите 'Дать рекомендации' или 'Пропустить' с помощью кнопок.")

    elif state == 'gpt_chat':
        if text == "завершить диалог":
            bot.send_message(chat_id,
                             "Диалог завершен. Если хотите начать новый анализ, введите /start",
                             reply_markup=remove_keyboard())
            del user_data[chat_id]
        else:
            # Добавляем сообщение пользователя в историю диалога
            user_data[chat_id]['gpt_history'].append({"role": "user", "content": m.text})
            response = openai.chat.completions.create(
                model=MODEL,
                messages=user_data[chat_id]['gpt_history'],
                max_tokens=500,
                temperature=0.7
            )
            gpt_response = response.choices[0].message.content.strip()
            user_data[chat_id]['gpt_history'].append({"role": "assistant", "content": gpt_response})
            bot.send_message(chat_id, gpt_response, reply_markup=create_gpt_chat_keyboard())
    else:
        bot.send_message(chat_id, "Неизвестное состояние. Начните с /start")

def send_prediction(chat_id):
    try:
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        user_input_df = pd.DataFrame([{
            'age': user_data[chat_id]['age'],
            'sex': user_data[chat_id]['sex'],
            'cp': user_data[chat_id]['cp'],
            'trestbps': user_data[chat_id]['trestbps'],
            'chol': user_data[chat_id]['chol'],
            'fbs': user_data[chat_id]['fbs'],
            'restecg': user_data[chat_id]['restecg'],
            'thalach': user_data[chat_id]['thalach'],
            'exang': user_data[chat_id]['exang'],
            'oldpeak': user_data[chat_id]['oldpeak'],
            'slope': user_data[chat_id]['slope'],
            'ca': user_data[chat_id]['ca'],
            'thal': user_data[chat_id]['thal']
        }], columns=columns)

        prediction_proba = model.predict_proba(user_input_df)[0][1] * 100

        # Сохраняем вероятность для последующего использования в рекомендациях
        user_data[chat_id]['prediction_proba'] = prediction_proba

        if prediction_proba > 50:
            bot.send_message(chat_id,
                             f"Вероятность сердечного заболевания по оценке модели: {prediction_proba:.2f}%. Рекомендуем обратиться к врачу.",
                             reply_markup=create_recommendation_keyboard())
        else:
            bot.send_message(chat_id,
                             f"Вероятность сердечного заболевания невысока ({prediction_proba:.2f}%).",
                             reply_markup=create_recommendation_keyboard())

        user_data[chat_id]['state'] = 'after_prediction'

    except Exception as e:
        bot.send_message(chat_id, f"Ошибка: {e}")

def get_recommendations(chat_id):
    data = user_data[chat_id]
    user_data_text = (
        f"Возраст: {data['age']}, Пол: {'Мужской' if data['sex'] == 1 else 'Женский'}, "
        f"Боль в груди: {data['cp']}, Давление: {data['trestbps']}, Холестерин: {data['chol']}, "
        f"Сахар натощак >120: {'Да' if data['fbs'] == 1 else 'Нет'}, ЭКГ: {data['restecg']}, "
        f"ЧСС: {data['thalach']}, Стенокардия: {'Да' if data['exang'] == 1 else 'Нет'}, "
        f"Oldpeak: {data['oldpeak']}, Наклон ST: {data['slope']}, Сосуды: {data['ca']}, Талассемия: {data['thal']}, "
        f"Вероятность болезни от модели: {data['prediction_proba']:.2f}%."
    )

    system_content = ("Ты опытный кардиолог с 20-летним стажем. "
                      "По данным пациента дай медицинские рекомендации по здоровому образу жизни и профилактике сердечно-сосудистых заболеваний. "
                      "Пиши ясным и простым языком, от первого лица, без маркированных списков.")
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_data_text}
    ]

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def get_clarification(question, chat_id):
    data = user_data[chat_id]
    user_data_text = (
        f"Возраст: {data['age']}, Пол: {'Мужской' if data['sex'] == 1 else 'Женский'}, "
        f"Боль в груди: {data['cp']}, Давление: {data['trestbps']}, Холестерин: {data['chol']}, "
        f"Сахар натощак >120: {'Да' if data['fbs'] == 1 else 'Нет'}, ЭКГ: {data['restecg']}, "
        f"ЧСС: {data['thalach']}, Стенокардия: {'Да' if data['exang'] == 1 else 'Нет'}, "
        f"Oldpeak: {data['oldpeak']}, Наклон ST: {data['slope']}, Сосуды: {data['ca']}, Талассемия: {data['thal']}, "
        f"Вероятность болезни: {data['prediction_proba']:.2f}%."
    )

    system_content = ("Ты опытный кардиолог с 20-летним стажем. Ранее ты дал рекомендации по профилактике сердечно-сосудистых заболеваний. "
                      "Теперь пациент задает уточняющий вопрос. Дай максимально понятный и полезный ответ, без списков и markdown.")
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_data_text},
        {"role": "user", "content": f"Вопрос: {question}"}
    ]

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def safe_polling():
    retry_interval = 5
    while True:
        try:
            bot.polling()
        except Exception as e:
            logging.error(e)
            time.sleep(retry_interval)
            retry_interval = min(retry_interval * 2, 60)

safe_polling()
