import json
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from types import TracebackType
from typing import Any

import requests

from .logger import get_logger

## TODO: data model for config - might use pydantic-settings


class NotificationSender:
    def __init__(
        self,
        email_config: dict | None = None,
        sms_config: dict | None = None,
        discord_config: dict | None = None,
        telegram_config: dict | None = None,
        wechat_config: dict | None = None,
        firebase_config: dict | None = None,
        logger: Any | None = None,
    ) -> None:
        self.email_config = email_config
        self.sms_config = sms_config
        self.discord_config = discord_config
        self.telegram_config = telegram_config
        self.wechat_config = wechat_config
        self.firebase_config = firebase_config
        self.session = requests.Session()

        self.logger = get_logger() if logger is None else logger

    def close(self) -> None:
        """Close the requests session and release resources."""
        if self.session:
            self.session.close()

    def __enter__(self) -> "NotificationSender":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.close()

    def send_email(self, subject: str, body: str, recipient: list[str]) -> None:
        if not self.email_config:
            self.logger.error("Not config on Email notification")
            return

        sender_email: str = self.email_config["sender_email"]
        username: str = self.email_config["username"]
        password: str = self.email_config["password"]
        smtp_server: str = self.email_config["smtp_server"]
        smtp_port: int = self.email_config.get("smtp_port", 587)

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = str(Header(subject, "utf-8"))
        msg["From"] = sender_email
        if isinstance(recipient, str):
            recipient = [recipient]
        msg["To"] = ", ".join(recipient)

        try:
            if smtp_port == 465:
                # with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                #     server.login(username if username else sender_email, password)
                #     result = server.sendmail(sender_email, recipient, msg.as_string())
                #     print(f'>>>> send_email 465 result: {result}')
                server: smtplib.SMTP_SSL | smtplib.SMTP = smtplib.SMTP_SSL(smtp_server, smtp_port)
            else:
                # with smtplib.SMTP(smtp_server, smtp_port) as server:
                #     if smtp_port == 587:
                #         server.starttls()
                #     server.login(username if username else sender_email, password)
                #     result = server.sendmail(sender_email, recipient, msg.as_string())
                #     print(f'>>>> send_email 587 result: {result}')
                server = smtplib.SMTP(smtp_server, smtp_port)
                if smtp_port == 587:
                    server.starttls()
            server.login(username if username else sender_email, password)
            server.sendmail(sender_email, recipient, msg.as_string())
            server.close()
            # self.logger.info("Email notification sent.")
        except Exception as e:
            self.logger.error(
                f"Email notification failed. sender_email:{sender_email}; smtp_server:{smtp_server}; smtp_port:{smtp_port}. Error:{str(e)}"
            )

    def send_discord(
        self,
        message: str | None = None,
        embeds: list[dict[Any, Any]] | dict[Any, Any] | None = None,
        avatar_url: str | None = None,
        username: str | None = None,
    ) -> None:
        if not self.discord_config:
            self.logger.error("Not config on Discord notification")
            return

        url = self.discord_config.get("webhook_url")

        headers = {"Content-type": "application/json"}

        payload: dict[str, Any] = {}
        if username:
            payload["username"] = username
        if message:
            payload["content"] = message
        if embeds:
            payload["embeds"] = embeds if isinstance(embeds, list) else [embeds]
        if avatar_url:
            payload["avatar_url"] = avatar_url

        try:
            response = self.session.post(url, headers=headers, json=payload)  # type: ignore
            if response.status_code == 204:
                # self.logger.info("Discord notification sent.")
                pass
            else:
                self.logger.error(f"Discord notification failed. response:{response.text}. payload:{payload}")
        except Exception as e:
            self.logger.error(f"Discord notification failed. Error:{str(e)}. payload:{payload}")

    def send_telegram(
        self,
        message: str,
        chat_ids: list[str] | str = [],
        disable_web_page_preview: bool = True,
        thread_id: str | None = None,
    ) -> None:
        """
        https://core.telegram.org/bots/api#sendmessage

        - Get bot_token >>>> https://gist.github.com/nafiesl/4ad622f344cd1dc3bb1ecbe468ff9f8a
        - Get chat_id >>>> https://api.telegram.org/bot{bot_token}/getUpdates
        - Rate limit: 30 messages per second & 20 messages per minute to the same group
        """
        if not self.telegram_config:
            self.logger.error("Not config on Telegram notification")
            return

        bot_token = self.telegram_config.get("bot_token")
        if isinstance(chat_ids, str):
            chat_ids = chat_ids.split(",")
        if not chat_ids:
            if self.telegram_config.get("chat_id"):
                if isinstance(self.telegram_config["chat_id"], str):
                    chat_ids = self.telegram_config["chat_id"].split(",")
                else:
                    chat_ids = [self.telegram_config["chat_id"]]
            else:
                self.logger.error("Not config on Telegram notification")
                return
        if not thread_id:
            thread_id = self.telegram_config.get("thread_id")

        for chat_id in chat_ids:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            params = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": disable_web_page_preview,
                "message_thread_id": thread_id,
            }

            try:
                response = self.session.post(url, params=params)  # type: ignore
                if response.status_code == 200:
                    # self.logger.info("Telegram notification sent.")
                    pass
                else:
                    self.logger.error(
                        f"Telegram notification failed to chat_id:{chat_id}. {response.text}. Msg:{message}"
                    )
            except Exception as e:
                self.logger.error(f"Telegram notification failed to chat_id:{chat_id}. Error:{str(e)}. Msg:{message}")

    def send_wechat_message(self, access_token: str, touser: str, message: str) -> None:
        url = f"https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={access_token}"
        data = {"touser": touser, "msgtype": "text", "text": {"content": message}}

        try:
            response = self.session.post(url, json=data)
            if response.status_code == 200:
                self.logger.info("WeChat message sent successfully.")
            else:
                self.logger.info("WeChat message failed to send.")
        except Exception as e:
            self.logger.error(f"WeChat notification failed. Error:{str(e)}")

    def send_qywechat(self, message: str) -> None:
        if not self.wechat_config:
            self.logger.error("Not config on Wechat notification")
            return

        corpid = self.wechat_config.get("corpid")
        corpsecret = self.wechat_config.get("corpsecret")
        agentid = self.wechat_config.get("agentid")
        touser = self.wechat_config.get("touser")

        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={corpid}&corpsecret={corpsecret}"
        response = self.session.get(url)
        access_token = response.json().get("access_token")

        url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}"
        data = {
            "touser": touser,
            "msgtype": "text",
            "agentid": agentid,
            "text": {"content": message},
            "safe": 0,
        }

        try:
            response = self.session.post(url, json=data)
            if response.status_code == 200:
                self.logger.info("Wechat notification sent.")
            else:
                self.logger.error("Wechat notification failed.")
        except Exception as e:
            self.logger.error("Wechat notification failed.", e)

    def send_firebase(self, message: str, topic: str | None = None, is_ola: bool = False) -> None:
        if not self.firebase_config:
            self.logger.error("Not config on Firebase notification")
            return

        import firebase_admin  # type: ignore
        from firebase_admin import credentials, messaging  # type: ignore

        cred = credentials.Certificate(self.firebase_config.get("firebase_key_path"))
        firebase_admin.initialize_app(cred)

        try:
            message_obj = messaging.Message(
                notification=messaging.Notification(title=message, body=message),
                topic=topic,
            )
            response = messaging.send(message_obj)
            self.logger.info("Firebase notification sent.", response)
        except Exception as e:
            self.logger.error("Firebase notification failed.", e)

    def send_sms(self, message: str, recipient: str) -> None:
        if not self.sms_config:
            self.logger.error("Not config on SMS notification")
            return

        account_sid = self.sms_config.get("account_sid")
        auth_token = self.sms_config.get("auth_token")
        from_number = self.sms_config.get("from_number")
        to_number = recipient

        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        params = {"From": from_number, "To": to_number, "Body": message}

        try:
            response = self.session.post(url, params=params, auth=(account_sid, auth_token))  # type: ignore
            if response.status_code == 201:
                self.logger.info("SMS notification sent.")
            else:
                self.logger.error("SMS notification failed.")
        except Exception as e:
            self.logger.error("SMS notification failed.", e)
