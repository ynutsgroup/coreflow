#!/usr/bin/env python3
from telegram import Update
from telegram.ext import Updater, CommandHandler

def start(update: Update, context):
    update.message.reply_text("Bot ist online!")

updater = Updater("DEIN_BOT_TOKEN")
updater.dispatcher.add_handler(CommandHandler("start", start))
updater.start_polling()
updater.idle()
