import time

def fileAppendTextLine(msg):
    with open('output.txt', 'a', encoding='utf-8') as file:
        file.write(f'[{time.time()}] {msg}\n')

if __name__ == '__main__':
    while True:
        fileAppendTextLine('test')
        time.sleep(1)