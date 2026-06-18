import os
import requests
from datetime import datetime

class AliyunSMS:
    def __init__(self):
        try:
            from aliyunsdkcore.client import AcsClient
            from aliyunsdkcore.profile import region_provider
            from aliyunsdkdysmsapi.request.v20170525 import SendSmsRequest
            from aliyunsdkcore.http import format_type
            self.available = True
            self.AcsClient = AcsClient
            self.region_provider = region_provider
            self.SendSmsRequest = SendSmsRequest
            self.format_type = format_type
        except ImportError:
            self.available = False
        
        self.access_key_id = os.environ.get("ALIYUN_ACCESS_KEY_ID", "")
        self.access_key_secret = os.environ.get("ALIYUN_ACCESS_KEY_SECRET", "")
        self.sign_name = os.environ.get("ALIYUN_SMS_SIGN_NAME", "")
        self.template_code = os.environ.get("ALIYUN_SMS_TEMPLATE_CODE", "")
        self.client = None
        
        if self.available and self.access_key_id and self.access_key_secret:
            try:
                self.client = self.AcsClient(self.access_key_id, self.access_key_secret, "cn-hangzhou")
                self.region_provider.add_endpoint("Dysmsapi", "cn-hangzhou", "dysmsapi.aliyuncs.com")
                print("✅ 阿里云短信客户端初始化成功")
            except Exception as e:
                print(f"❌ 阿里云短信客户端初始化失败: {e}")
                self.client = None
    
    def send(self, phone_number: str, message: str) -> dict:
        if not self.client:
            return {"success": False, "message": "阿里云短信未配置"}
        
        try:
            request = self.SendSmsRequest.SendSmsRequest()
            request.set_accept_format(self.format_type.JSON)
            request.set_PhoneNumbers(phone_number)
            request.set_SignName(self.sign_name)
            request.set_TemplateCode(self.template_code)
            request.set_TemplateParam(f'{{"message":"{message}"}}')
            
            response = self.client.do_action_with_exception(request)
            result = eval(response.decode('utf-8'))
            
            if result.get("Code") == "OK":
                return {"success": True, "message": "发送成功", "biz_id": result.get("BizId")}
            else:
                return {"success": False, "message": result.get("Message")}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def is_configured(self) -> bool:
        return self.client is not None

class TencentSMS:
    def __init__(self):
        try:
            from qcloudsms_py import SmsSingleSender
            self.available = True
            self.SmsSingleSender = SmsSingleSender
        except ImportError:
            self.available = False
        
        self.app_id = os.environ.get("TENCENT_APP_ID", "")
        self.app_key = os.environ.get("TENCENT_APP_KEY", "")
        self.sign_name = os.environ.get("TENCENT_SMS_SIGN_NAME", "")
        self.template_id = int(os.environ.get("TENCENT_SMS_TEMPLATE_ID", "0"))
    
    def send(self, phone_number: str, message: str) -> dict:
        if not self.available or not self.app_id or not self.app_key:
            return {"success": False, "message": "腾讯云短信未配置"}
        
        try:
            ssender = self.SmsSingleSender(int(self.app_id), self.app_key)
            params = [message]
            result = ssender.send_with_param(
                86, phone_number, self.template_id, params, 
                sign=self.sign_name, extend="", ext=""
            )
            
            if result.get("result") == 0:
                return {"success": True, "message": "发送成功"}
            else:
                return {"success": False, "message": result.get("errmsg", "发送失败")}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def is_configured(self) -> bool:
        return self.available and self.app_id and self.app_key

class HuaweiSMS:
    def __init__(self):
        self.endpoint = "https://rtcsms.cn-north-1.myhuaweicloud.com:10743"
        self.app_key = os.environ.get("HUAWEI_APP_KEY", "")
        self.app_secret = os.environ.get("HUAWEI_APP_SECRET", "")
        self.sign_channel = os.environ.get("HUAWEI_SIGN_CHANNEL", "")
    
    def send(self, phone_number: str, message: str) -> dict:
        if not self.app_key or not self.app_secret:
            return {"success": False, "message": "华为云短信未配置"}
        
        try:
            token = self._get_token()
            if not token:
                return {"success": False, "message": "获取token失败"}
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            
            data = {
                "from": self.sign_channel,
                "to": [phone_number],
                "templateId": "SMS_001",
                "templateParas": [message]
            }
            
            response = requests.post(f"{self.endpoint}/sms/batchSend", headers=headers, json=data)
            result = response.json()
            
            if result.get("code") == "000000":
                return {"success": True, "message": "发送成功"}
            else:
                return {"success": False, "message": result.get("description", "发送失败")}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def _get_token(self):
        try:
            response = requests.post(
                "https://oauth.cn-north-1.myhuaweicloud.com/v3/auth/tokens",
                json={
                    "auth": {
                        "identity": {
                            "methods": ["password"],
                            "password": {
                                "user": {
                                    "name": self.app_key,
                                    "password": self.app_secret,
                                    "domain": {"name": "hw00000000"}
                                }
                            }
                        },
                        "scope": {"project": {"name": "cn-north-1"}}
                    }
                }
            )
            return response.headers.get("X-Subject-Token")
        except:
            return None
    
    def is_configured(self) -> bool:
        return self.app_key and self.app_secret

class RonglianSMS:
    def __init__(self):
        self.account_sid = os.environ.get("RONGLIAN_ACCOUNT_SID", "")
        self.auth_token = os.environ.get("RONGLIAN_AUTH_TOKEN", "")
        self.app_id = os.environ.get("RONGLIAN_APP_ID", "")
        self.server_url = "https://app.cloopen.com:8883/2013-12-26"
    
    def send(self, phone_number: str, message: str) -> dict:
        if not self.account_sid or not self.auth_token or not self.app_id:
            return {"success": False, "message": "容联云短信未配置"}
        
        try:
            import hashlib
            import time
            
            timestamp = time.strftime("%Y%m%d%H%M%S")
            sig = self.account_sid + self.auth_token + timestamp
            sig = hashlib.md5(sig.encode()).hexdigest().upper()
            
            url = f"{self.server_url}/Accounts/{self.account_sid}/SMS/TemplateSMS?sig={sig}"
            
            headers = {
                "Content-Type": "application/json;charset=utf-8",
                "Authorization": self._get_auth(self.account_sid, timestamp)
            }
            
            data = {
                "to": phone_number,
                "appId": self.app_id,
                "templateId": "1",
                "datas": [message]
            }
            
            response = requests.post(url, headers=headers, json=data)
            result = response.json()
            
            if result.get("statusCode") == "000000":
                return {"success": True, "message": "发送成功"}
            else:
                return {"success": False, "message": result.get("statusMsg", "发送失败")}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def _get_auth(self, account_sid, timestamp):
        import base64
        auth_str = f"{account_sid}:{timestamp}"
        return base64.b64encode(auth_str.encode()).decode()
    
    def is_configured(self) -> bool:
        return self.account_sid and self.auth_token and self.app_id

class HuYiSMS:
    def __init__(self):
        self.account = os.environ.get("HUYI_ACCOUNT", "")
        self.password = os.environ.get("HUYI_PASSWORD", "")
        self.server_url = "http://106.ihuyi.com/webservice/sms.php?method=Submit"
    
    def send(self, phone_number: str, message: str) -> dict:
        if not self.account or not self.password:
            return {"success": False, "message": "互亿无线短信未配置"}
        
        try:
            data = {
                "account": self.account,
                "password": self.password,
                "mobile": phone_number,
                "content": f"【智能视频告警】{message}",
                "format": "json"
            }
            
            response = requests.post(self.server_url, data=data)
            result = response.json()
            
            if result.get("code") == 2:
                return {"success": True, "message": "发送成功"}
            else:
                return {"success": False, "message": result.get("msg", "发送失败")}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def is_configured(self) -> bool:
        return self.account and self.password

class WeComBot:
    def __init__(self):
        self.webhook_url = os.environ.get("WECOM_WEBHOOK_URL", "")
        self.mentioned_mobiles = os.environ.get("WECOM_MENTIONED_MOBILES", "").split(",")
    
    def send(self, message: str) -> dict:
        if not self.webhook_url:
            return {"success": False, "message": "企业微信机器人未配置"}
        
        try:
            data = {
                "msgtype": "text",
                "text": {
                    "content": f"🚨 智能视频告警\n\n{message}",
                    "mentioned_mobile_list": [m.strip() for m in self.mentioned_mobiles if m.strip()]
                }
            }
            
            response = requests.post(self.webhook_url, json=data)
            result = response.json()
            
            if result.get("errcode") == 0:
                return {"success": True, "message": "企业微信消息发送成功"}
            else:
                return {"success": False, "message": result.get("errmsg", "发送失败")}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def is_configured(self) -> bool:
        return bool(self.webhook_url)

class DingTalkBot:
    def __init__(self):
        self.webhook_url = os.environ.get("DINGTALK_WEBHOOK_URL", "")
        self.secret = os.environ.get("DINGTALK_SECRET", "")
    
    def send(self, message: str) -> dict:
        if not self.webhook_url:
            return {"success": False, "message": "钉钉机器人未配置"}
        
        try:
            import time
            import hmac
            import hashlib
            import base64
            
            timestamp = str(round(time.time() * 1000))
            
            if self.secret:
                string_to_sign = f"{timestamp}\n{self.secret}"
                hmac_code = hmac.new(
                    self.secret.encode(), 
                    string_to_sign.encode(), 
                    digestmod=hashlib.sha256
                ).digest()
                sign = base64.b64encode(hmac_code).decode()
                url = f"{self.webhook_url}&timestamp={timestamp}&sign={sign}"
            else:
                url = self.webhook_url
            
            data = {
                "msgtype": "text",
                "text": {
                    "content": f"🚨 智能视频告警\n\n{message}"
                }
            }
            
            response = requests.post(url, json=data)
            result = response.json()
            
            if result.get("errcode") == 0:
                return {"success": True, "message": "钉钉消息发送成功"}
            else:
                return {"success": False, "message": result.get("errmsg", "发送失败")}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def is_configured(self) -> bool:
        return bool(self.webhook_url)

class EmailNotifier:
    def __init__(self):
        self.smtp_server = os.environ.get("EMAIL_SMTP_SERVER", "smtp.qq.com")
        self.smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", "465"))
        self.smtp_user = os.environ.get("EMAIL_SMTP_USER", "")
        self.smtp_password = os.environ.get("EMAIL_SMTP_PASSWORD", "")
        self.to_emails = os.environ.get("EMAIL_TO", "").split(",")
    
    def send(self, message: str) -> dict:
        if not self.smtp_user or not self.smtp_password:
            return {"success": False, "message": "邮件服务未配置"}
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.header import Header
            
            msg = MIMEText(f"<h2>🚨 智能视频告警</h2><p>{message}</p>", "html", "utf-8")
            msg["From"] = Header("智能视频告警系统", "utf-8")
            msg["To"] = Header(",".join(self.to_emails), "utf-8")
            msg["Subject"] = Header("【告警】智能视频检测到异常事件", "utf-8")
            
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, [e.strip() for e in self.to_emails if e.strip()], msg.as_string())
            
            return {"success": True, "message": "邮件发送成功"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def is_configured(self) -> bool:
        return bool(self.smtp_user) and bool(self.smtp_password)

class SMSManager:
    def __init__(self):
        self.sms_providers = {
            "aliyun": AliyunSMS(),
            "tencent": TencentSMS(),
            "huawei": HuaweiSMS(),
            "ronglian": RonglianSMS(),
            "huyi": HuYiSMS()
        }
        
        self.notify_providers = {
            "wecom": WeComBot(),
            "dingtalk": DingTalkBot(),
            "email": EmailNotifier()
        }
        
        self.active_sms_provider = self._detect_active_sms_provider()
        self.active_notify_providers = self._detect_active_notify_providers()
    
    def _detect_active_sms_provider(self):
        for name, provider in self.sms_providers.items():
            if provider.is_configured():
                print(f"✅ 检测到已配置的短信服务商: {name}")
                return provider
        print("⚠️ 未检测到已配置的短信服务商")
        return None
    
    def _detect_active_notify_providers(self):
        active = []
        for name, provider in self.notify_providers.items():
            if provider.is_configured():
                print(f"✅ 检测到已配置的通知方式: {name}")
                active.append(provider)
        if not active:
            print("⚠️ 未检测到已配置的通知方式")
        return active
    
    def send_sms(self, phone_number: str, message: str) -> dict:
        results = []
        
        if self.active_sms_provider:
            result = self.active_sms_provider.send(phone_number, message)
            results.append({"type": "sms", "phone": phone_number, **result})
            self.log_notification("sms", phone_number, message, result["success"], result.get("message"))
        
        for provider in self.active_notify_providers:
            result = provider.send(message)
            provider_name = type(provider).__name__.replace("Bot", "").replace("Notifier", "").lower()
            results.append({"type": provider_name, **result})
            self.log_notification(provider_name, "-", message, result["success"], result.get("message"))
        
        if not results:
            return {"success": False, "message": "未配置任何通知方式"}
        
        return {
            "success": any(r["success"] for r in results),
            "message": "通知发送完成",
            "details": results
        }
    
    def log_notification(self, notify_type: str, target: str, message: str, success: bool, error_msg: str = ""):
        log_dir = "./notifications"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"notifications_{datetime.now().strftime('%Y%m%d')}.log")
        status = "SUCCESS" if success else "FAILED"
        
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {notify_type.upper()} | {status} | {target} | {message[:100]}")
            if error_msg:
                f.write(f" | {error_msg}")
            f.write("\n")
    
    def is_configured(self) -> bool:
        return self.active_sms_provider is not None or len(self.active_notify_providers) > 0
    
    def get_providers(self):
        return list(self.sms_providers.keys()) + list(self.notify_providers.keys())

sms_manager = SMSManager()

def send_alarm_sms(phone_number: str, alarm_info: dict) -> dict:
    """发送告警短信"""
    if not sms_manager.is_configured():
        return {"success": False, "message": "短信服务未配置"}
    
    message = f"{alarm_info.get('message', '未知告警')} 告警ID:{alarm_info.get('id', '')}"
    return sms_manager.send_sms(phone_number, message)