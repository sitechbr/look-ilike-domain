from rapidfuzz.fuzz import ratio, partial_ratio, token_sort_ratio
from ahocorasick import Automaton

def build_automaton(domains):
    automaton = Automaton()
    for domain in domains:
        clean_domain = domain.strip().lower()
        automaton.add_word(clean_domain, clean_domain)
    automaton.make_automaton()
    return automaton

def find_similar_domains(input_domain, domains, threshold=85):
    input_domain = input_domain.lower()
    similar_domains = []
    for domain in domains:
        domain_lower = domain.lower()
        if ratio(input_domain, domain_lower) >= threshold:
            similar_domains.append(domain)
    return similar_domains

def check_domain(input_domain, automaton, domains, threshold=50):
    input_domain = input_domain.lower()

    # Aho-Corasick match
    for end_index, match in automaton.iter(input_domain):
        return True, match, 100

    # Fuzzy matching
    similar_domains = find_similar_domains(input_domain, domains, threshold)
    if similar_domains:
        best_match = max(similar_domains, key=lambda d: ratio(input_domain, d.lower()))
        score = ratio(input_domain, best_match.lower())
        return True, best_match, score

    return False, None, 0

if __name__ == "__main__":
    with open('data/emails.txt', 'r') as file:
        domains = [line.strip() for line in file if line.strip()]
    
    automaton = build_automaton(domains)

    # Примеры входных доменов
    test_domains = [
        "google.com",       # точное совпадение
        "g00gle.com",        # замена букв
        "goog1e.com",        # замена на похожий символ
        "yahooo.com",        # добавление буквы
        "faceb00k.com",      # заменённые символы
        "xn--80ak6aa92e.com" # Punycode (не обрабатывается без нормализации!)
    ]

    for test_domain in test_domains:
        is_lookalike, matched, score = check_domain(test_domain, automaton, domains)
        print(f"{test_domain:20} => Lookalike: {is_lookalike}, Match: {matched}, Score: {score}")
