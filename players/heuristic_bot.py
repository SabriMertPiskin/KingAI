import random

class HeuristicBot:
    def __init__(self, name="RuleBot"):
        self.name = name

    def choose_move(self, legal_moves, engine, player_idx):
        """
        Basit Kurallar:
        1. Asla elindeki Kız'ı yere atma (Eğer mecbur değilsen).
        2. El sendeyse (yer boşsa) en küçük kağıdını çık.
        3. El başkasındaysa ve sende o renkten yoksa (çakacaksan), Kız veya As çak.
        4. El başkasındaysa ve o renkten varsa, yerdeki en büyük kağıdı geçmemeye çalış.
        """
        # Kartları (Suit, Rank) olarak analiz et
        # Rank: 11=J, 12=Q, 13=K, 14=A
        
        # 1. MECBURİYET DURUMLARI
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Yerde kağıt var mı?
        trick_cards = [c[1] for c in engine.current_trick]
        
        # --- SEN BAŞLIYORSAN (LEAD) ---
        if not trick_cards:
            # Strateji: En küçük kağıdını oyna ama Kız (12) veya Papaz (13) olmasın
            safe_moves = [c for c in legal_moves if c[1] < 12]
            if safe_moves:
                # En küçüğünü bul
                return min(safe_moves, key=lambda x: x[1])
            else:
                # Hepsi büyükse yapacak bir şey yok, en küçüğünü at
                return min(legal_moves, key=lambda x: x[1])

        # --- TAKİP EDİYORSAN (FOLLOW) ---
        first_suit = trick_cards[0][0]
        my_suit_moves = [c for c in legal_moves if c[0] == first_suit]
        
        # Durum A: O renk bende YOK (Çakma Fırsatı!)
        if not my_suit_moves:
            # Varsa Kız (12) at, yoksa Papaz/As at (Yüksekten kurtul)
            queens = [c for c in legal_moves if c[1] == 12]
            if queens: return queens[0]
            
            high_cards = [c for c in legal_moves if c[1] > 10]
            if high_cards: return max(high_cards, key=lambda x: x[1])
            
            return max(legal_moves, key=lambda x: x[1])

        # Durum B: O renk bende VAR (Mecburiyet)
        else:
            # Yerdeki en yüksek o renk kağıdı bul
            current_winner_card = max([c for c in trick_cards if c[0] == first_suit], key=lambda x: x[1], default=None)
            current_max_rank = current_winner_card[1] if current_winner_card else 0
            
            # Yerde Kız var mı?
            queen_in_trick = any(c[1] == 12 for c in trick_cards)
            
            if queen_in_trick:
                # Yerde Kız var! Ne yap et bu eli alma.
                # Yerdekinden küçük atabiliyor muyum?
                smaller_moves = [c for c in legal_moves if c[1] < current_max_rank]
                if smaller_moves:
                    return max(smaller_moves, key=lambda x: x[1]) # En büyüğün küçüğünü at
                else:
                    # Mecbur alacağız, en büyüğü at bari diğerleri de ezilsin (veya en küçüğü at zarar ziyan olmasın)
                    return max(legal_moves, key=lambda x: x[1])
            else:
                # Yerde kız yok.
                # Eğer elimde Kız varsa ve onu atmak zorundaysam (tek çaremse) yapacak bir şey yok.
                # Ama seçeneğim varsa Kız'ı (12) sakla.
                safe_moves = [c for c in legal_moves if c[1] != 12]
                if not safe_moves: safe_moves = legal_moves
                
                # Yerdeki kağıdı geçmemeye çalış (Under-cut)
                under_moves = [c for c in safe_moves if c[1] < current_max_rank]
                if under_moves:
                    return max(under_moves, key=lambda x: x[1])
                
                return min(safe_moves, key=lambda x: x[1])