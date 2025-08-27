# -*- coding: utf-8 -*-
"""
Typhoon Day-off LINE Bot 
--------------------------------------------

What this bot does:
1) Receives a text message from a user in LINE.
2) If the text matches a region keyword (e.g., "北部"), replies with quick-reply buttons for cities.
3) If the text matches a city (e.g., "@臺北市"), replies with quick-reply buttons for districts mapped to a **CWB station**.
4) If the text matches a **station name** (e.g., "臺北", "淡水", ...), it:
   - Fetches live observations from the **CWB Open Data** API (that is your "crawling" step).
   - Engineers features (humidity %, wind/gust vectors).
   - Imputes missing values with your **KNN imputer** (joblib artifact).
   - Adds (currently hard-coded) **typhoon meta features** (e.g., hpa, TyWS, route).
   - Scales features with your **MinMaxScaler** (joblib artifact).
   - Predicts the probability of **tomorrow's day-off** with your **RandomForest** model (joblib artifact).
   - Sends a friendly message based on the probability range.

Security note (tokens/keys):
- The script contains hard-coded **LINE tokens** and a **CWB API key**. For production/GitHub, consider reading them
  from environment variables (e.g., os.getenv) and storing real values in a `.env` file or secret manager.
"""
#Import library
from multiprocessing.sharedctypes import Value
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, StickerSendMessage, LocationSendMessage, QuickReply, QuickReplyButton, MessageAction
import joblib
import sklearn
import numpy as np
import os
import requests
import json
import urllib
import pandas as pd
from linebot.exceptions import InvalidSignatureError
from linebot import LineBotApi, WebhookHandler
from flask import Flask, request, abort

# Flask app: webhook receiver for LINE
app = Flask(__name__)
# LINE channel access token (consider loading from env in production)
line_bot_api = LineBotApi(
    'GypJ9iKN7FFxN8O9lzEHBvRnRU/P//kFgIUmQD5mUoxrv2EpoSqxlp/CCAfh2lP7EEUgPD3ch4rFzQPZtZ5w4G+YEiMohMSGRPkvYYrjKbSgVYzPMwCSNvczn5tJpP+f9Jf/7hCK3CGwXm1YhcmGuQdB04t89/1O/w1cDnyilFU=')
# LINE channel secret (consider loading from env in production)
handler = WebhookHandler('ece7c238e5be91ce40171bd08a6c6d4b')

# ===== WEBHOOK ENTRYPOINT =====
@app.route("/callback", methods=['POST'])
def callback():
    """LINE webhook endpoint: verifies signature and dispatches to handle_message via WebhookHandler."""
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# Acceptable station names
locations = ['基隆', '彭佳嶼', '國三N002K', '國三S004K', '國三S007K', '國一S001K', '國一S006K', '竹子湖', '鞍部', '臺北', '大安森林', '國三甲005K', '國三S016K', '板橋', '淡水', '新店', '國三S010K', '國三S037K', '國三S042K', '國三N046K', '國三S054K', '國一S026K', '拉拉山', '新屋', '國二E009K', '國三N063K', '國一S072K', '西濱S023K', '西濱S032K', '國三S103K', '西濱S082K', '新竹', '國一S105K', '國三N076K', '國一N077K', '西濱N066K', '國一N142K', '國一S114K', '國一S123K', '國一S132K', '國一S152K', '國三N119K', '國三N151K', '國三S140K', '西濱N107K', '臺中', '梧棲', '武陵', '國一N174K', '國一S162K', '國一S169K', '國一S188K', '國三N208K', '國三S173K', '國三S178K', '西濱S189K',
             '西濱N210K', '彰師大', '田中', '國一S207K', '國三N191K', '日月潭', '玉山', '國三N223K', '國三S217K', '麥寮', '古坑', '國一N234K', '國三N252K', '西濱S241K', '西濱N222K', '國三N295K', '嘉義', '阿里山', '國一N250K', '國一S262K', '國一N268K', '國三N303K', '西濱S267K', '西濱S257K', '南區中心', '永康', '國一N288K', '國一S306K', '國一N316K', '國一N335K', '國三S311K', '國三S329K', '國三S342K', '國三S366K', '西濱S280K', '西濱N304K', '高雄', '東沙島', '國一S342K', '國一S346K', '國一N351K', '國一N361K', '國三N372K', '國三S385K', '恆春', '國三N425K', '宜蘭', '蘇澳', '花蓮', '合歡山', '成功', '臺東', '大武', '蘭嶼', '吉貝', '東吉島', '澎湖', '金門(東)', '金門', '九宮碼頭', '東引', '馬祖']

# ===== LINE BOT UI: REGION → CITY → STATION =====
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """Main message handler: handles region -> city -> station flow; runs CWB fetch + ML pipeline on station input."""
    mtext = event.message.text
    # 撰寫接受到的選項對應的動作
    # Incoming user text
    # --- Region selectors: return city-level quick-replies ---
    if mtext == '北部':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的縣市',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="基隆市", text="@基隆市")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="臺北市", text="@臺北市")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新北市", text="@新北市")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="桃園市", text="@桃園市")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新竹市", text="@新竹市")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新竹縣", text="@新竹縣")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="苗栗縣", text="@苗栗縣")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '中部':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的縣市',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="臺中市", text="@臺中市")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="彰化縣", text="@彰化縣")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="南投縣", text="@南投縣")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="雲林縣", text="@雲林縣")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="嘉義市", text="@嘉義市")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新竹縣", text="@新竹縣")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="嘉義縣", text="@嘉義縣")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '南部':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的縣市',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="臺南市", text="@臺南市")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="高雄市", text="@高雄市")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="屏東縣", text="@屏東縣")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '東部':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的縣市',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="宜蘭縣", text="@宜蘭縣")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="花蓮縣", text="@花蓮縣")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="臺東縣", text="@臺東縣")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '外島':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的縣市',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="澎湖縣", text="@澎湖縣")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="金門縣", text="@金門縣")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="連江縣", text="@連江縣")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@基隆市':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="仁愛區", text="基隆")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="中正區", text="彭佳嶼")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="七堵區", text="國三S004K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="安樂區", text="國一S001K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))
    elif mtext == '@臺北市':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="北投區", text="竹子湖")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="中正區", text="臺北")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="大安區", text="大安森林")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="文山區", text="國三甲005K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="南港區", text="國三S016K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@新北市':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="板橋區", text="板橋")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="淡水區", text="淡水")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新店區", text="新店")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="汐止區", text="國三S010K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="土城區", text="國三S037K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="樹林區", text="國三N046K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="鶯歌區", text="國三S054K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="三重區", text="國一S026K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@桃園市':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="復興區", text="拉拉山")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新屋區", text="新屋")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="桃園區", text="國二E009K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="大溪區", text="國三N063K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="楊梅區", text="國一S072K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="蘆竹區", text="西濱S023K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="大園區", text="西濱S032K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@新竹市':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="東區", text="國三S103K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="香山區", text="西濱S082K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@新竹縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="竹北市", text="新竹")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="寶山鄉", text="國一S105K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="關西鎮", text="國三N076K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="湖口鄉", text="國一N077K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新豐鄉", text="西濱N066K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@苗栗縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="銅鑼鄉", text="國一N142K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="頭份市", text="國一S114K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="頭屋鄉", text="國一S123K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="公館鄉", text="國一S132K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="三義鄉", text="國一S152K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="竹南鎮", text="國三N119K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="苑裡鎮", text="國三N151K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="通霄鎮", text="國三S140K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="後龍鎮", text="西濱N107K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@臺中市':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="北區", text="臺中")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="梧棲區", text="梧棲")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="和平區", text="武陵")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="西屯區", text="國一N174K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="后里區", text="國一S162K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="神岡區", text="國一S169K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="大肚區", text="國一S188K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="霧峰區", text="國三N208K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="清水區", text="國三S173K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="沙鹿區", text="國三S178K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@彰化縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="芳苑鄉", text="西濱S189K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="大城鄉", text="西濱N210K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="彰化市", text="彰師大")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="田中鎮", text="田中")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="大村鄉", text="國一S207K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="和美鎮", text="國三N191K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@南投縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="魚池鄉", text="日月潭")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="信義鄉", text="玉山")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="草屯鎮", text="國三N223K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@雲林縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="麥寮鄉", text="麥寮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="古坑鄉", text="古坑")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="西螺鎮", text="國一N234K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="口湖鄉", text="西濱S241K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@嘉義市':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="東區", text="國三N295K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="西區", text="嘉義")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@嘉義縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="阿里山鄉", text="阿里山")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="大林鎮", text="國一N250K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="民雄鄉", text="國一S262K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水上鄉", text="國一N268K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="中埔鄉", text="國三N303K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="東石鄉", text="西濱S267K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@臺南市':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="中西區", text="南區中心")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="永康區", text="永康")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新營區", text="國一N288K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="麻豆區", text="國一S306K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新市區", text="國一N316K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="仁德區", text="國一N335K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="白河區", text="國三S311K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="官田區", text="國三S329K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="善化區", text="國三S342K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="關廟區", text="國三S366K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="北門區", text="西濱S280K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="七股區", text="西濱N304K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@高雄市':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="前鎮區", text="高雄")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="旗津區", text="東沙島")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="路竹區", text="國一S342K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="岡山區", text="國一S346K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="橋頭區", text="國一N351K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="仁武區", text="國一N361K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="田寮區", text="國三N372K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="大樹區", text="國三S385K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@屏東縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="竹田鄉", text="國三S415K")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="恆春鎮", text="恆春")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="南州鄉", text="國三N425K")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@宜蘭縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="宜蘭市", text="宜蘭")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="蘇澳鎮", text="蘇澳")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@花蓮縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="花蓮市", text="花蓮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="秀林鄉", text="合歡山")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@臺東縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="成功鎮", text="成功")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="臺東市", text="臺東")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="大武鄉", text="大武")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="蘭嶼鄉", text="蘭嶼")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@澎湖縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="白沙鄉", text="吉貝")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="望安鄉", text="東吉島")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="馬公市", text="澎湖")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@金門縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="金湖鎮", text="金門(東)")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="金城鎮", text="金門")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="烈嶼鄉", text="九宮碼頭")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))

    elif mtext == '@連江縣':
        try:
            message = TextSendMessage(
                text='請選擇欲觀測的地區，將幫您對應到該地區測站',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="東引鄉", text="東引")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="南竿鄉", text="馬祖")
                        )
                    ]
                )
            )
            line_bot_api.reply_message(event.reply_token, message)
        except:
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text='發生錯誤!'))
# ===== STATION → PIPELINE: CRAWLING → PROCESSING → PREDICTING =====
    elif mtext in locations:
        locationName = mtext
        try:
            # --- Crawling (CWB API) ---
            df_get = get(locationName)  # 獲得天氣data
            data1 = get_datashow(locationName)
            # --- Processing: Imputation (if NaNs present) ---
            df_imputed = df_get.copy()
            if df_get.isna().sum().sum() != 0:  # 判斷是否需要補值
                kNN_imputer = joblib.load(os.getenv('KNN_IMPUTER_PATH', 'models/kNN_imputer.joblib'))
                imputed_values = kNN_imputer.transform(df_get)
                df_imputed = pd.DataFrame(imputed_values, columns = df_get.columns)
            # --- Processing: Add Typhoon Meta (static placeholders) ---
            df_full = merge_typhoon_data(df_imputed)  # 合併
            # --- Processing: Scaling (MinMax) ---
            MMscaler = joblib.load(os.getenv('MINMAX_SCALER_PATH', 'models/MMscaler.joblib'))
            try:
                from src.predict import SCALER_COLS
                x = MMscaler.transform(df_full[SCALER_COLS])
            except Exception:
                x = MMscaler.transform(df_full)
            # --- Predicting: RandomForest Proba ---
            model = joblib.load(os.getenv('MODEL_PATH', 'models/rf_model.joblib'))
            prediction = model.predict_proba(x)
            dayoff_proba = prediction[0][1]*100
            # UI: graded messages by probability range
            if dayoff_proba >= 90:
                try:
                    message = [
                        TextSendMessage(
                            text = "目前氣溫" + data1.iat[0,3]+"度"
                        ),
                        TextSendMessage(
                            text = "累積雨量" + data1.iat[0,6]+"mm"
                        ),
                        TextSendMessage(
                            text = "天氣" + data1.iat[0,20]
                        ),
                        TextSendMessage(
                            text=f'明天放颱風假機率:{round(dayoff_proba, 1)}%\n超高機率放颱風假，祝您假期愉快！'
                        )
                    ]
                except:
                    message = TextSendMessage(
                        text="不好意思~請您再試一次"
                    )
            elif dayoff_proba >= 80:
                try:
                    message = [
                        TextSendMessage(
                            text = "目前氣溫" + data1.iat[0,3]+"度"
                        ),
                        TextSendMessage(
                            text = "累積雨量" + data1.iat[0,6]+"mm"
                        ),
                        TextSendMessage(
                            text = "天氣" + data1.iat[0,20]
                        ),
                        TextSendMessage(
                            text=f'明天放颱風假機率:{round(dayoff_proba, 1)}%\n高機率放颱風假，請做好準備！'
                        )
                    ]
                except:
                    message = TextSendMessage(
                        text="不好意思~請您再試一次"
                    )
            elif dayoff_proba >= 60:
                try:
                    message = [
                        TextSendMessage(
                            text = "目前氣溫" + data1.iat[0,3]+"度"
                        ),
                        TextSendMessage(
                            text = "累積雨量" + data1.iat[0,6]+"mm"
                        ),
                        TextSendMessage(
                            text = "天氣" + data1.iat[0,20]
                        ),
                        TextSendMessage(
                            text=f'明天放颱風假機率:{round(dayoff_proba, 1)}%\n可以期待一下颱風假！'
                        )
                    ]
                except:
                    message = TextSendMessage(
                        text="不好意思~請您再試一次"
                    )
            elif dayoff_proba >= 40:
                try:
                    message = [
                        TextSendMessage(
                            text = "目前氣溫" + data1.iat[0,3]+"度"
                        ),
                        TextSendMessage(
                            text = "累積雨量" + data1.iat[0,6]+"mm"
                        ),
                        TextSendMessage(
                            text = "天氣" + data1.iat[0,20]
                        ),
                        TextSendMessage(
                            text=f'明天放颱風假機率:{round(dayoff_proba, 1)}%\n預言大師也算不準明天到底會不會放颱風假...'
                        )
                    ]
                except:
                    message = TextSendMessage(
                        text="不好意思~請您再試一次"
                    )
            elif dayoff_proba >= 20:
                try:
                    message = [
                        TextSendMessage(
                            text = "目前氣溫" + data1.iat[0,3]+"度"
                        ),
                        TextSendMessage(
                            text = "累積雨量" + data1.iat[0,6]+"mm"
                        ),
                        TextSendMessage(
                            text = "天氣" + data1.iat[0,20]
                        ),
                        TextSendMessage(
                            text=f'明天放颱風假機率:{round(dayoff_proba, 1)}%\n明天不太可能放颱風假哦！'
                        )
                    ]
                except:
                    message = TextSendMessage(
                        text="不好意思~請您再試一次"
                    )
            else:
                try:
                    message = [
                        TextSendMessage(
                            text = "目前氣溫" + data1.iat[0,3]+"度"
                        ),
                        TextSendMessage(
                            text = "累積雨量" + data1.iat[0,6]+"mm"
                        ),
                        TextSendMessage(
                            text = "天氣" + data1.iat[0,20]
                        ),
                        TextSendMessage(
                            text=f'明天放颱風假機率:{round(dayoff_proba, 1)}%\n您還是別妄想颱風假了～～'
                        )
                    ]
                except:
                    message = TextSendMessage(
                        text="不好意思~請您再試一次"
                    )

            line_bot_api.reply_message(event.reply_token, message)
        except Exception:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text='服務暫時出現問題，請稍後再試'))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text='請從「北部/中部/南部/東部/外島」開始'))

# ===== CRAWLING: Station Observation (ML features) =====
def get(locationName):
    """Fetch live observation from CWB and engineer ML features (RH %, wind/gust vectors). Returns 1-row DataFrame."""
    url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/O-A0003-001?'
    params = {
        'Authorization': 'CWB-5393733F-1F8E-4BD2-A358-0360AABEE6EB',
        'format': 'JSON',
        'locationName': locationName,
    }

    url = url + urllib.parse.urlencode(params)
    data = requests.get(url, timeout=10).json()
    stnInfo = data['records']['location'][0]
    weatherElement = data['records']['location'][0]['weatherElement']
    obs = {
        'lat': [float(stnInfo['lat'])],
        'lon': [float(stnInfo['lon'])],
    }
    name_mapping = {
        'lat': 'lat',
        'lon': 'lon',
        'Dayoff': 'Dayoff',
        'ELEV': 'StnHeight',
        'HUMD': 'RH',
        'D_TX': 'T.Max',
        'D_TN': 'T.Min',
        'TEMP': 'Temperature',
        'PRES': 'StnPres',
        '24R': 'Precp',
        'WD_vector_x': 'WD_vector_x',
        'WD_vector_y': 'WD_vector_y',
        'WDGust_vector_x': 'WDGust_vector_x',
        'WDGust_vector_y': 'WDGust_vector_y',
        'WDSD': 'A',
        'WDIR': 'B',
        'H_XD': 'C',
        'H_FX': 'D',

    }
    for i in range(len(weatherElement)):  ###########
        name = weatherElement[i]['elementName']
        value = weatherElement[i]['elementValue']
        if name in name_mapping.keys():
            obs[name] = [float(value)]
            if float(value) < -90:
                obs[name] = [np.nan]
            else:
                if name == 'HUMD':
                    v = float(value)*100
                    obs[name] = [v]

        # 處理wind
        # 轉換 WS, WG, WSGust, WDGust
        # 氣象角度轉為及座標單位向量公式：(-theta + 90) * pi/180，再分別以cos, sin處理得到單位向量
        # 風速x單位向量
    # Wind vectors    
    try:
        WD = obs['WDIR'][0]
        WS = obs['WDSD'][0]
        WD_unit_vector_x = round(np.cos((-WD + 90) * np.pi/180), 5)
        WD_unit_vector_y = round(np.sin((-WD + 90) * np.pi/180), 5)
        WD_vector_x = np.sqrt(WS) * WD_unit_vector_x
        WD_vector_y = np.sqrt(WS) * WD_unit_vector_y
    except:
        WD_vector_x = np.nan
        WD_vector_y = np.nan

    try:
        WDGust = obs['H_XD'][0]
        WSGust = obs['H_FX'][0]
        WDGust_unit_vector_x = round(np.cos((-WDGust + 90) * np.pi/180), 5)
        WDGust_unit_vector_y = round(np.sin((-WDGust + 90) * np.pi/180), 5)
        WDGust_vector_x = np.sqrt(WSGust) * WDGust_unit_vector_x
        WDGust_vector_y = np.sqrt(WSGust) * WDGust_unit_vector_y
    except:
        WDGust_vector_x = np.nan
        WDGust_vector_y = np.nan

    obs['WD_vector_x'] = WD_vector_x
    obs['WD_vector_y'] = WD_vector_y
    obs['WDGust_vector_x'] = WDGust_vector_x
    obs['WDGust_vector_y'] = WDGust_vector_y

    df = pd.DataFrame(obs)
    df = df.rename(name_mapping, axis=1)
    df = df.drop(['A', 'B', 'C', 'D', ], axis=1)
    df = df.reindex(sorted(df.columns), axis = 1)
    return df
# ===== CRAWLING: Human-readable display =====
def get_datashow(locationName):
    """Fetch human-readable CWB observation table for direct texting (e.g., temperature string)."""
    url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/O-A0003-001?'
    params = {
        'Authorization': 'CWB-5393733F-1F8E-4BD2-A358-0360AABEE6EB', # NOTE: move to env var in production
        'format': 'JSON',
        'locationName': locationName,
    }

    url = url + urllib.parse.urlencode(params)
    data1 = requests.get(url, timeout=10).json()
    data1 = pd.DataFrame(data1['records']['location'][0]['weatherElement']).T
    data1.columns = data1.loc['elementName',:]
    data1 = data1.drop('elementName', axis = 0)
    return data1

# def kNN_imputation(df):
#     # kNN補值，若沒有NA則
#     kNN_imputer = joblib.load(os.getcwd()+'/models/kNN_imputer.joblib')
#     df = kNN_impoter.fit_transform(df)
#     return df

# ===== PROCESSING: Typhoon meta features (static placeholders) =====
def merge_typhoon_data(df_get):
    """Attach static typhoon meta features (e.g., hpa, TyWS, route_*). Keep columns aligned with scaler."""   
 #    颱風的資料寫死在這裡，選的是莫拉克颱風
  #  1. 字典形式列出特徵工程後還存在的颱風參數
 #   2. 合併抓下來的觀測資料與颱風資料
 #   3. 回傳含有颱風&觀測資料的dataframe
    # 1.
    typhoon_feature = {
        'Dayoff': [1.],  # 假設放假 the data with holiday
        'route_3': [1.],
        'route_2': [1.],
        'route_--': [1.],
        'hpa': [955.],
        'TyWS': [40.],
        'X7_radius': [250.],
        'X10_radius': [100.],
        'alert_num': [36.],
        'born_spotE': [136.],
        'born_spotN': [21.],
    }

    df_typhoon = pd.DataFrame(typhoon_feature)
    df_get = df_get.join(df_typhoon, how='left')
    df_get = df_get.reindex(sorted(df_get.columns), axis=1)
    return df_get


# def MM_scale(df):
#     # 使用fit training set的MinMaxScaler壓縮data到0~1之間
#     MMscaler = joblib.load(os.getcwd()+'/models/MMscaler.joblib')
#     df_MM = MMscaler.transform(df)

#     return df_MM


# def prediction(x):
#     # 使用model進行預測
#     # 要改成mtext回覆的格式
#     model = joblib.load(os.getcwd()+'/models/rf_model.joblib')
#     prediction = model.predict_proba(x)
#     dayoff_proba = prediction[0][1]*100
#     return dayoff_proba


# ===== SERVER START =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
