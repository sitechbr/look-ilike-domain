import ahocorasick
from rapidfuzz import fuzz
from typing import List, Tuple, Optional
import unicodedata
import os

class LookalikeDomainDetector:
    def __init__(self, legitimate_domains_file: str = "data/email.txt"):
        # Инициализируем словарь замены символов в первую очередь
        self.similar_chars = {
            '0': 'o', '1': 'l', 'i': 'l', '5': 's', '2': 'z', '3': 'e', '4': 'a',
            'о': 'o', 'а': 'a', 'е': 'e', 'с': 'c', 'ӏ': 'l', 'р': 'p'
        }
        
        if not os.path.exists(legitimate_domains_file):
            raise FileNotFoundError(f"Файл {legitimate_domains_file} не найден")
        
        self.legitimate_domains = self._load_legitimate_domains(legitimate_domains_file)
        self.automaton = self._build_aho_corasick_automaton()

    def _load_legitimate_domains(self, file_path: str) -> List[str]:
        domains = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '@' in line:
                    domain = line.split('@')[1].lower()
                    domains.add(domain)
        return list(domains)

    def _build_aho_corasick_automaton(self) -> ahocorasick.Automaton:
        automaton = ahocorasick.Automaton()
        for domain in self.legitimate_domains:
            automaton.add_word(domain, domain)
            normalized = self._normalize(domain)
            if normalized != domain:
                automaton.add_word(normalized, domain)
        automaton.make_automaton()
        return automaton

    def _normalize(self, text: str) -> str:
        if not hasattr(self, 'similar_chars'):
            return text.lower()
            
        text = text.lower()
        text = ''.join(self.similar_chars.get(c, c) for c in text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        return text

    def _find_similar_domains(self, domain: str) -> List[str]:
        normalized = self._normalize(domain)
        similar = set()
        
        for _, legit_domain in self.automaton.iter(normalized):
            similar.add(legit_domain)
            
        if len(normalized) <= 12:
            for legit_domain in self.legitimate_domains:
                if fuzz.partial_ratio(normalized, self._normalize(legit_domain)) > 80:
                    similar.add(legit_domain)
                    
        return list(similar)

    def _calculate_similarity(self, domain1: str, domain2: str) -> float:
        norm1 = self._normalize(domain1)
        norm2 = self._normalize(domain2)
        ratio = fuzz.ratio(norm1, norm2) / 100
        partial = fuzz.partial_ratio(norm1, norm2) / 100
        token = fuzz.token_sort_ratio(norm1, norm2) / 100
        return max(ratio, partial, token)

    def analyze_domain(self, domain: str, threshold: float = 0.5) -> Tuple[bool, Optional[str], float]:
        if not domain:
            return False, None, 0.0

        if domain in self.legitimate_domains:
            return False, None, 1.0

        similar_domains = self._find_similar_domains(domain)
        if not similar_domains:
            return False, None, 0.0

        best_match = None
        best_score = 0.0
        
        for legit_domain in similar_domains:
            score = self._calculate_similarity(domain, legit_domain)
            if score > best_score:
                best_score = score
                best_match = legit_domain

        return best_score >= threshold, best_match, best_score

def test_detector():
    test_data = [
        ("gmail.com", False),
        ("gmail.c0m", True),
        ("gmail.comm", True),
        ("gmai1.com", True),
        ("yahoo.com", False),
        ("yaho0.com", True),
        ("protonmail.org", False),
        ("pr0tonmail.org", True),
        ("google.com", False),
        ("goog1e.com", True),
        ("bankofamerica.com", False),
        ("bankofamer1ca.com", True),
        ("example.com", False),
        ("xn--80ak6aa92e.com", False),
        ("аррӏе.com", True)
    ]

    os.makedirs("data", exist_ok=True)
    with open("data/email.txt", "w", encoding='utf-8') as f:
        f.write("user@gmail.com\n")
        f.write("admin@yahoo.com\n")
        f.write("support@protonmail.org\n")
        f.write("contact@google.com\n")
        f.write("info@bankofamerica.com\n")
        f.write("help@apple.com\n")

    detector = LookalikeDomainDetector()

    print("Результаты тестирования:")
    print("-" * 60)
    for domain, expected in test_data:
        is_lookalike, original, score = detector.analyze_domain(domain)
        result = "✅" if is_lookalike == expected else "❌"
        print(f"{result} {domain.ljust(20)} -> "
              f"Lookalike: {is_lookalike} (expected: {expected}), "
              f"Original: {original or 'N/A'}, "
              f"Score: {score:.2f}")

if __name__ == "__main__":
    test_detector()