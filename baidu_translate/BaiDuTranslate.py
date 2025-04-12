import string
import hashlib
import requests
import random
from urllib import parse


class BaiDuTranslate:

    def __init__(self, appid, key, fromLan='auto', toLan='zh'):
        self.appid = appid
        self.key = key
        self.fromLan = fromLan
        self.toLan = toLan

    # md5加密
    def md5Encryption(self, text):
        hashl = hashlib.md5()
        hashl.update(text.encode(encoding='utf8'))
        secret_key = hashl.hexdigest()
        return secret_key

    # 递归查找目标key
    def get_key(self, js_data, target_key, results=[]):
        if isinstance(js_data, dict):
            for key in js_data.keys():
                data = js_data[key]
                self.get_key(data, target_key, results=results)
                if key == target_key:
                    results.append(data)
        elif isinstance(js_data, list) or isinstance(js_data, tuple):
            for data in js_data:
                self.get_key(data, target_key, results=results)
        return results

    def create_alt(self):
        return str(random.randint(32768, 65536))

    def create_sign(self, q):
        alt = self.create_alt()
        str_ = self.appid + q + alt + self.key
        sign = self.md5Encryption(str_)
        return alt, sign

    # 请求api，返回翻译结果
    def requestApi(self, query):
        translateApi = 'https://fanyi-api.baidu.com/api/trans/vip/translate?'
        salt, sign = self.create_sign(query)
        url = (translateApi + 'q=' + parse.quote(query, encoding='utf-8') +
               '&from=' + self.fromLan +
               '&to=' + self.toLan +
               '&appid=' + self.appid +
               '&salt=' + salt +
               '&sign=' + sign)

        re = requests.get(url)
        data = re.json()
        result = ''.join(self.get_key(js_data=data, target_key='dst', results=[]))
        
        # print("目标语言:", self.toLan)
        # print("翻译结果:", result)

        return result

if __name__ == "__main__":
    q = 'ゲームの話じゃなくて、元になった猫の妖精のこと'
    key = 'key'
    appid = 'api'

    translator = BaiDuTranslate(key=key, appid=appid, toLan='en')
    translate_result = translator.requestApi(query=q)
    print("最终翻译结果:", translate_result)
