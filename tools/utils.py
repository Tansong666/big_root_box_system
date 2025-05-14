import socket
import ipaddress
import time

def get_local_ip():
    """
    获取本机所有IP
    :return: IP列表
    """
    local_ips = ["127.0.0.1"]
    for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
        local_ips.append(ip)

    local_ips.sort(reverse=True)
    return local_ips


def decode_data(bytes_arr: bytes) -> str:
    """自动尝试不同的字符集进行解码

    :param bytes_arr: 字节数组
    :return: 字符串
    """
    try:
        msg = bytes_arr.decode("UTF-8")
    except Exception as e:
        msg = bytes_arr.decode("GBK")
        
    return msg


def decode_to_hex_str(data: bytes):
    # 将hex字符串转成大写，每两个字符用空格分隔
    data_hex = data.hex().upper()
    data_hex = " ".join([data_hex[i:i + 2] for i in range(0, len(data_hex), 2)])
    return data_hex


class Color:
    """
    颜色枚举类
    """
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    PURPLE = 35
    CYAN = 36


def wrap_color(msg, color):
    """
    包装颜色
    :param msg: 要包装的字符串
    :param color: 颜色
    :return:
    """
    return f"\033[{color}m{msg}\033[0m"


def calculate_broadcast_address(host, mask_bits):
    # 将主机IP地址和掩码位数转换为网络对象
    network = ipaddress.ip_network(f"{host}/{mask_bits}", strict=False)

    # 获取广播地址
    broadcast_address = network.broadcast_address

    # 返回广播地址的字符串表示形式
    return str(broadcast_address)


def current_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


if __name__ == '__main__':
    print(get_local_ip())
    
    print(wrap_color("hello world", Color.GREEN))