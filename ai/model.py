import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KingDQN(nn.Module):
    def __init__(self, input_size=230, output_size=52): 
        super(KingDQN, self).__init__()
        
        # --- DEĞİŞİKLİK BURADA ---
        # BatchNorm yerine LayerNorm kullanıyoruz.
        # LayerNorm, input dimension üzerinden normalize eder, batch size 1 olsa bile çalışır.
        
        self.fc1 = nn.Linear(input_size, 512)
        self.ln1 = nn.LayerNorm(512)  # BatchNorm1d -> LayerNorm
        
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)  # BatchNorm1d -> LayerNorm
        
        self.fc3 = nn.Linear(512, 256)
        # Çıkış katmanından önce genelde Norm kullanılmaz, relu yeterli.
        
        self.head = nn.Linear(256, output_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Eğer tekli veri gelirse (1D), onu 2D (Batch=1) yap
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # --- DEĞİŞİKLİK BURADA ---
        # LayerNorm kullanımı:
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        
        return self.head(x).squeeze(0)

# get_state_vector — V2: Zenginleştirilmiş state vektörü
def get_state_vector(engine, player_idx):
    state = []
    
    # --- 1. HAM KART BİLGİLERİ (One-Hot) ---
    my_hand = set(engine.hands[player_idx])
    deck_order = [(s, r) for s in engine.suits for r in engine.ranks]
    
    # 1a. Benim elim (52)
    hand_vec = [1.0 if c in my_hand else 0.0 for c in deck_order]
    state.extend(hand_vec)

    # 1b. Masadaki kartlar (52)
    trick_cards = [c[1] for c in engine.current_trick]
    trick_vec = [1.0 if c in trick_cards else 0.0 for c in deck_order]
    state.extend(trick_vec)

    # 1c. Daha önce oynanan kartlar (52) — BUG FIX: artık doğru çalışıyor!
    played_vec = [1.0 if c in engine.played_cards else 0.0 for c in deck_order]
    state.extend(played_vec)

    # --- 2. STRATEJİK FEATURELAR ---
    # 2a. Açılan renk bilgisi (4 + 1 + 1 = 6)
    if engine.current_trick:
        first_suit = engine.current_trick[0][1][0]
        suit_map = {'Spade':0, 'Heart':1, 'Diamond':2, 'Club':3}
        trick_suit_vec = [0.0]*4
        if first_suit in suit_map:
            trick_suit_vec[suit_map[first_suit]] = 1.0
        state.extend(trick_suit_vec)
        
        same_suit_cards = [c[1] for c in engine.current_trick if c[1][0] == first_suit]
        if same_suit_cards:
            max_rank = max(c[1] for c in same_suit_cards)
            state.append(max_rank / 14.0)
        else:
            state.append(0.0)
            
        has_suit = any(c[0] == first_suit for c in engine.hands[player_idx])
        state.append(1.0 if has_suit else 0.0)
    else:
        state.extend([0.0]*4)
        state.append(0.0)
        state.append(0.0)

    # 2b. Kız takibi — her Kız nerede? (4 × 3 = 12)
    queens = [('Spade', 12), ('Heart', 12), ('Diamond', 12), ('Club', 12)]
    for q in queens:
        if q in my_hand:
            state.extend([0.0, 1.0, 0.0])     # Benim elimde
        elif q in trick_cards or q in engine.played_cards:
            state.extend([0.0, 0.0, 1.0])     # Oynandı / masada
        else:
            state.extend([1.0, 0.0, 0.0])     # Rakipte (tehlike!)

    # 2c. Elimdeki renk sayıları (4)
    counts = {'Spade':0, 'Heart':0, 'Diamond':0, 'Club':0}
    for c in engine.hands[player_idx]:
        counts[c[0]] += 1
    state.extend([counts[s]/13.0 for s in ['Spade', 'Heart', 'Diamond', 'Club']])

    # 2d. Elde kalan kart sayısı (1)
    state.append(len(engine.hands[player_idx]) / 13.0)
    
    # 2e. Benim cezam (1)
    state.append(engine.penalties[player_idx] / -100.0)
    
    # ═══ YENİ FEATURELAR ═══
    
    # 3a. Rakiplerin void suit bilgisi (3 rakip × 4 renk = 12)
    #     Hangi rakibin hangi rengi bittiğini bilmek kız çakma stratejisi için kritik
    opponents = [i for i in range(4) if i != player_idx]
    for opp in opponents:
        if hasattr(engine, 'void_suits'):
            state.extend([1.0 if engine.void_suits[opp][s] else 0.0 for s in range(4)])
        else:
            state.extend([0.0]*4)
    
    # 3b. Oyunda kaç Kız hâlâ oynamamış? (1)
    queens_remaining = 0
    for q in queens:
        if q not in engine.played_cards and q not in trick_cards:
            queens_remaining += 1
    state.append(queens_remaining / 4.0)
    
    # 3c. Kaçıncı el — oyun fazı farkındalığı (1)
    if hasattr(engine, 'trick_number'):
        state.append(engine.trick_number / 13.0)
    else:
        state.append(0.0)
    
    # 3d. Rakip cezaları (3)
    for opp in opponents:
        state.append(engine.penalties[opp] / -100.0)
    
    # Padding — input_size=230'a sabit tamamla
    while len(state) < 230:
        state.append(0.0)
        
    return np.array(state[:230], dtype=np.float32)