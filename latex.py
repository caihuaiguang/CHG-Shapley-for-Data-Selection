import re

# 原始文本
text = """
    & AdaptiveRandom
    & 0.7364 & 0.82& 0.8873 & 0.8961 & 0.8992 & 0.3645 & 0.4944& 0.6138 & 0.6344 & 0.6427  \\\\
    & Glister
    & 0.678 & 0.8147& 0.8316 & 0.8207 & 0.8863 & 0.3279 &0.4815& 0.6086 &0.6194 & 0.6272  \\\\
    & GradMatchPB
    & 0.7512 & 0.8292& 0.8821 & 0.8955 & 0.8971 & 0.3945 & 0.5139& 0.6154 &  0.6397 & 0.6407  \\\\
    & Hardness Shapley
    & 0.3557 & 0.6495& 0.8795 & 0.8913 & 0.8956 &  0.2587 & 0.4334& 0.6142 &  0.6343 & 0.6431\\\\
    & TracIn (Gradient-Dot)
    & 0.4297 & 0.6834& 0.8845 & 0.8942 & 0.8954 & 0.3181 & 0.4352& 0.611 & 0.6354 & 0.6402    \\\\
    & CGSV (Gradient-Cosine)
    & 0.6309 & 0.7445& 0.876 & 0.8972 &  0.8901 & 0.3235 & 0.4258& 0.5803 & 0.63 & 0.6438    \\\\
    & Gradient Shapley (This paper)
    & 0.6761 & 0.8098& 0.8646 & 0.8755 & 0.8855 &  0.3965 & 0.4896& 0.5781 & 0.6102 & 0.6214   \\\\
    & CHG Shapley (This paper)
    & 0.6994 & 0.8365& 0.8769 & 0.8861 & 0.8886 & 0.4031 & 0.5188& 0.6033 & 0.629 & 0.6389  \\\\
"""

# 转换函数
# def convert_numbers(text):
#     def replace(match):
#         number = int(match.group())
#         return str(round(number /3600, 1))

#     # 使用正则表达式替换数字
#     return re.sub(r'\b\d+\b', replace, text)

# 转换函数
def convert_numbers(text):
    def replace(match):
        number = float(match.group())
        return str(round(number * 100, 2))

    # 使用正则表达式替换数字
    return re.sub(r'\b\d+\.\d+\b', replace, text)


# 执行转换
converted_text = convert_numbers(text)

# 打印结果
print(converted_text)
