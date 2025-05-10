import random

# Чтение данных из файлов
with open('lists/names.txt', 'r') as f:
    names = [line.strip() for line in f if line.strip()]

with open('lists/domains.txt', 'r') as f:
    domains = [line.strip() for line in f if line.strip()]

with open('lists/tids.txt', 'r') as f:
    tids = [line.strip() for line in f if line.strip()]

# Проверка, что файлы не пустые
if not names or not domains or not tids:
    raise ValueError("Один или несколько файлов пусты!")

# Генерация 100 email-адресов
emails = []
for _ in range(100000):
    name = random.choice(names)
    domain = random.choice(domains)
    tid = random.choice(tids)
    email = f"{name}@{domain}.{tid}"
    emails.append(email)

# Сохранение в файл
with open('emails.txt', 'w') as f:
    for email in emails:
        f.write(email + '\n')

print(f"Сгенерировано {len(emails)} email-адресов в emails.txt")