import os
import datetime
import logging
import pickle
from enum import Enum
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)


class AstrologicalSign(Enum):
    Aries = "Овен"
    Taurus = "Телец"
    Gemini = "Близнецы"
    Cancer = "Рак"
    Leo = "Лев"
    Virgo = "Дева"
    Libra = "Весы"
    Scorpio = "Скорпион"
    Sagittarius = "Стрелец"
    Capricorn = "Козерог"
    Aquarius = "Водолей"
    Pisces = "Рыбы"

    def get_names_with_emojis():
        emojis = ["♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓"]
        res = []

        emoji_index = 0
        for sign in AstrologicalSign:
            res.append(emojis[emoji_index] + " " + sign.value)
            emoji_index += 1

        return res


class HoroscopeGPTState(Enum):
    DEFAULT = 0
    WAITING_FOR_SIGN = 1
    WAITING_FOR_TIME = 2


class HoroscopeGPT:
    def __init__(self, temp=0.9, send_time=datetime.time(hour=4)):
        self.temp = temp
        self.send_time = send_time
        self.state = HoroscopeGPTState.DEFAULT
        self.load_users_signs()
        self.init_llm()
        self.init_bot()

    def load_users_signs(self):
        self.users_signs = dict()
        if os.path.exists("users_signs.pkl"):
            with open("users_signs.pkl", "rb") as f:
                self.users_signs = pickle.load(f)

    def dump_users_signs(self):
        with open("users_signs.pkl", "wb") as f:
            pickle.dump(self.users_signs, f)

    def init_llm(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            temperature=self.temp, model_name="gpt-3.5-turbo"
        )

    def init_bot(self):
        self.bot = ApplicationBuilder().token(os.getenv("TG_API_KEY")).build()

        handler = CommandHandler("start", self.start_handler)
        self.bot.add_handler(handler)

        handler = MessageHandler(filters.TEXT, self.text_handler)
        self.bot.add_handler(handler)

        logging.info(self.users_signs)
        for chat_id in self.users_signs:
            self.bot.job_queue.run_daily(
                self.send_horoscope_job,
                self.send_time,
                data=chat_id,
            )

    def run(self):
        self.bot.run_polling()

    async def get_horoscope(self, sign):
        prompt = (
            f"Напиши гороскоп для знака зодиака {sign.value} на "
            + f"{datetime.datetime.now().strftime('%d.%m.%Y')}. "
            + f"Используй не более 200 слов."
        )
        return self.llm.predict(prompt)

    def shape_keyboard(self, choices, per_row):
        keyboard = []
        row = []
        for choice in choices:
            if len(row) < per_row:
                row.append(choice)
            else:
                keyboard.append(row)
                row = [choice]
        return keyboard

    async def start_handler(self, update, context):
        reply_keyboard = self.shape_keyboard(
            AstrologicalSign.get_names_with_emojis(), 2
        )
        await context.bot.send_message(
            text="Выберите знак зодиака",
            chat_id=update.effective_chat.id,
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True
            ),
        )
        self.state = HoroscopeGPTState.WAITING_FOR_SIGN

    async def text_handler(self, update, context):
        if self.state == HoroscopeGPTState.DEFAULT:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Воспользуйтесь командой /start, если хотите изменить настройки",
            )
        elif self.state == HoroscopeGPTState.WAITING_FOR_SIGN:
            sign_name = update.message.text[2:]
            for sign in AstrologicalSign:
                if sign.value == sign_name:
                    if update.message.chat_id not in self.users_signs:
                        context.job_queue.run_daily(
                            self.send_horoscope_job,
                            self.send_time,
                            data=update.message.chat_id,
                        )
                    self.users_signs[update.message.chat_id] = sign
                    self.dump_users_signs()
                    context.job_queue.run_once(
                        self.send_horoscope_job, 0, data=update.message.chat_id
                    )
                    break
        elif self.state == HoroscopeGPTState.WAITING_FOR_TIME:
            pass
        self.state = HoroscopeGPTState.DEFAULT

    async def send_horoscope_job(self, context):
        sign = self.users_signs[context.job.data]
        horoscope = await self.get_horoscope(sign)
        await context.bot.send_message(
            chat_id=context.job.data, text=horoscope
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    app = HoroscopeGPT()
    app.run()
