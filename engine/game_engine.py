import random
import json
import os
import time

class KizAlmazEngine:
    def __init__(self):
        self.suits = ['Spade', 'Heart', 'Diamond', 'Club']
        self.ranks = list(range(2, 15)) # 11:J, 12:Q, 13:K, 14:A
        self.queens = [(s, 12) for s in self.suits]
        
        # Loglama Sistemi Başlangıcı
        self.game_log = {
            "initial_hands": [],
            "actions": [],   
            "tricks": [],    
            "final_scores": []
        }
        self.reset()

    def reset(self):
        self.hands = self._distribute_and_save()
        
        # Yeni oyun için logları sıfırla
        self.game_log["initial_hands"] = [h[:] for h in self.hands] 
        self.game_log["actions"] = []
        self.game_log["tricks"] = []
        
        self.played_cards = []
        self.current_trick = []
        self.penalties = [0, 0, 0, 0]
        self.turn = 0 
        self.game_over = False
        self.trick_number = 0  # Kaçıncı el (0-12)
        # void_suits[player][suit] = True → o oyuncunun o rengi bitmiş
        self.void_suits = [[False]*4 for _ in range(4)]

    def _distribute_and_save(self):
        deck = [(s, r) for s in self.suits for r in self.ranks]
        random.shuffle(deck)
        # Kartları sıralı tutmak strateji ve okunabilirlik için önemli
        hands = [sorted(deck[i:i+13], key=lambda x: (x[0], x[1])) for i in range(0, 52, 13)]
        
        os.makedirs('data', exist_ok=True)
        # UI için son dağıtımı tutan dosya
        with open("data/last_deal.json", "w") as f:
            json.dump({"hands": hands}, f)
        return hands

    def get_legal_moves(self, player_idx):
        """Oyuncunun elindeki yasal hamleleri döndürür."""
        hand = self.hands[player_idx]
        if not self.current_trick:
            return hand
        
        first_suit = self.current_trick[0][1][0]
        same_suit_cards = [c for c in hand if c[0] == first_suit]
        
        return same_suit_cards if same_suit_cards else hand

    def play_move(self, player_idx, card):
        """Bir kart oynar ve sırayı ilerletir."""
        
        # --- LOGLAMA: HAMLE ÖNCESİ DURUM ---
        log_entry = {
            "turn_idx": len(self.game_log["actions"]),
            "player": player_idx,
            "played_card": card,
            "table_cards": [c[1] for c in self.current_trick],
            "legal_moves": self.get_legal_moves(player_idx),
            "current_penalties": self.penalties[:]
        }
        self.game_log["actions"].append(log_entry)
        # -----------------------------------

        if card in self.hands[player_idx]:
            self.hands[player_idx].remove(card)
        
        # Void suit tracking: eğer açılan renkten oynamadıysa, o renk bitmiş
        if self.current_trick:  # Ben lead değilsem
            first_suit = self.current_trick[0][1][0]
            if card[0] != first_suit:
                suit_map = {'Spade':0, 'Heart':1, 'Diamond':2, 'Club':3}
                self.void_suits[player_idx][suit_map[first_suit]] = True
        
        self.current_trick.append((player_idx, card))
        
        # Eğer yerde 4 kart varsa eli bitir (Kazananı belirle)
        if len(self.current_trick) == 4:
            self._resolve_trick()
        else:
            # Eli bitmediyse sıra yanındakine geçer
            self.turn = (self.turn + 1) % 4

    def _resolve_trick(self):
        """Yerdeki 4 kağıdı değerlendirir, kazananı ve cezayı belirler."""
        first_card_suit = self.current_trick[0][1][0]
        
        # Kazananı bul (Rengi uyan en büyük kağıt)
        winner_tuple = max(
            [t for t in self.current_trick if t[1][0] == first_card_suit],
            key=lambda x: x[1][1]
        )
        winner_idx = winner_tuple[0]
        
        # Ceza kontrolü (Kız var mı?)
        points_taken = 0
        has_queen = False
        for _, card in self.current_trick:
            if card in self.queens:
                self.penalties[winner_idx] -= 20
                points_taken -= 20
                has_queen = True
        
        # --- LOGLAMA: EL SONUCU ---
        trick_log = {
            "trick_cards": [t[1] for t in self.current_trick],
            "winner": winner_idx,
            "points": points_taken,
            "has_queen": has_queen
        }
        self.game_log["tricks"].append(trick_log)
        # --------------------------

        # Masayı temizle, sırayı kazanana ver
        self.turn = winner_idx
        self.played_cards.extend([t[1] for t in self.current_trick])  # ÖNCE kaydet!
        self.current_trick = []  # SONRA temizle
        self.trick_number += 1  # El sayacı

        # --- FİNAL KONTROLÜ (BUG FIX) ---
        # Eğer oyuncuların elinde kart kalmadıysa oyunu bitir.
        if len(self.hands[0]) == 0:
            self.game_over = True
        # --------------------------------

    def save_full_log(self):
        """Oyun bittiğinde tüm veriyi diske yazar (UI kullanımı için)."""
        self.game_log["final_scores"] = self.penalties
        timestamp = int(time.time())
        filename = f"data/game_history_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(self.game_log, f, indent=4)
        return filename