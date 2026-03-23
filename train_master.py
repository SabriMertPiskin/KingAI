"""
train_master.py — Grandmaster Seviye AI Eğitimi
================================================
Temel fark: Kız çakma stratejisi öğreten gelişmiş reward shaping.

Yenilikler:
  1. Queen Dump Reward — Kızını rakibe çakınca +5 ödül
  2. Opponent Pain Reward — Rakip ceza yeyince +2 ödül
  3. Queen Holding Penalty — Geç turda elde Kız tutmak -1/Kız
  4. Void Creation Bonus — Renk boşaltınca +0.3 bonus
  5. Self-Play fazları — Kendi kopyasına karşı da oynar
  6. 10K episode, GTX 4070 optimized
"""

import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import os
import copy
from engine.game_engine import KizAlmazEngine
from ai.model import KingDQN, get_state_vector
from players.heuristic_bot import HeuristicBot
from ai.memory import ReplayBuffer

# ═══════════════════════════════════════════
#  HYPERPARAMETERS
# ═══════════════════════════════════════════
EPISODES       = 10_000
BATCH_SIZE     = 256
GAMMA          = 0.99
EPSILON_START  = 1.0
EPSILON_END    = 0.01
EPSILON_DECAY  = 5000
TARGET_UPDATE  = 30
LR             = 0.0001
MEMORY_SIZE    = 200_000
MODEL_SAVE     = "king_ai_master.pth"

# Faz geçişleri
PHASE_1_END = 3000    # Ep 1-3000:  vs HeuristicBot
PHASE_2_END = 8000    # Ep 3001-8000: vs Self-Play (kendi kopyası)
                      # Ep 8001-10000: Mixed (50/50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════
#  SELF-PLAY BOT (Eğitilen modelin kopyası)
# ═══════════════════════════════════════════
class SelfPlayBot:
    """Eğitilen policy net'in frozen bir kopyası ile oynayan bot."""
    def __init__(self, model, engine):
        self.model = model
        self.engine = engine
        self.deck_order = [(s, r) for s in engine.suits for r in engine.ranks]

    def choose_move(self, legal_moves, engine, player_idx):
        state = get_state_vector(engine, player_idx)
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            mask = torch.full(q_values.shape, float('-inf')).to(device)
            for m in legal_moves:
                idx = self.deck_order.index(m)
                mask[idx] = q_values[idx]
            action_idx = torch.argmax(mask).item()
        return self.deck_order[action_idx]


def get_legal_mask(engine, player_idx, deck_order):
    legal_moves = engine.get_legal_moves(player_idx)
    mask = np.zeros(52, dtype=np.float32)
    for move in legal_moves:
        idx = deck_order.index(move)
        mask[idx] = 1.0
    return mask


# ═══════════════════════════════════════════
#  REWARD HESAPLAMA
# ═══════════════════════════════════════════
def compute_reward(engine, player_idx, card_played, prev_ai_score, prev_opp_scores, trick_just_resolved):
    """
    Gelişmiş ödül fonksiyonu.
    trick_just_resolved: El az önce çözüldüyse True (4 kart tamamlandıysa).
    """
    reward = 0.0
    curr_ai_score = engine.penalties[player_idx]

    # ─── 1. TEMEL: Ceza yedim mi? ───
    ai_pain = curr_ai_score - prev_ai_score
    if ai_pain < 0:
        reward += -10.0   # Ceza yedim, çok kötü

    # ─── 2. QUEEN DUMP: Kızımı çaktım mı? ───
    if trick_just_resolved and card_played[1] == 12:
        # Kız attım, eli ben almadıysam → çaktım!
        if ai_pain == 0:
            reward += 5.0   # "Kızımı çaktım, rakip yedi!"

    # ─── 3. OPPONENT PAIN: Rakip ceza yedi mi? ───
    if trick_just_resolved:
        for opp in [1, 2, 3]:
            opp_pain = engine.penalties[opp] - prev_opp_scores[opp]
            if opp_pain < 0:
                reward += 2.0   # Her ceza yiyen rakip başına +2

    # ─── 4. QUEEN HOLDING: Geç turda elde Kız tutma cezası ───
    queens_in_hand = sum(1 for c in engine.hands[player_idx] if c[1] == 12)
    cards_left = len(engine.hands[player_idx])
    if queens_in_hand > 0 and cards_left <= 5:
        reward -= 1.0 * queens_in_hand

    # ─── 5. VOID CREATION: Renk boşaltma bonusu ───
    if cards_left > 0:
        suits_in_hand = len(set(c[0] for c in engine.hands[player_idx]))
        if suits_in_hand < 4:
            reward += 0.3 * (4 - suits_in_hand)

    # ─── 6. HAYATTA KALMA: Ceza yemedim, basit bonus ───
    if ai_pain == 0 and not trick_just_resolved:
        if card_played[1] >= 12:
            reward += 0.5   # Riskli kart güvenli çıktı
        else:
            reward += 0.1

    return reward


# ═══════════════════════════════════════════
#  ANA EĞİTİM DÖNGÜSÜ
# ═══════════════════════════════════════════
def train_master():
    engine = KizAlmazEngine()

    policy_net = KingDQN().to(device)
    target_net = KingDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Sıfırdan eğitim (warm start YOK — yeni state vector ile temiz başlangıç)
    print("⚡ Sıfırdan eğitim — yeni state vector ile temiz başlangıç!")

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE, device)

    heuristic_bots = [HeuristicBot(), HeuristicBot(), HeuristicBot()]
    deck_order = [(s, r) for s in engine.suits for r in engine.ranks]

    # Self-play kopyası (Faz 2+ için)
    selfplay_model = KingDQN().to(device)
    selfplay_model.load_state_dict(policy_net.state_dict())
    selfplay_model.eval()

    # Metrik takip
    running_reward = 0.0
    running_ai_penalty = 0.0
    queen_dumps = 0

    print(f"🔥 MASTER EĞİTİM BAŞLIYOR — {device} ({EPISODES} episode)")
    print(f"   Faz 1: Ep 1-{PHASE_1_END} (vs Heuristic)")
    print(f"   Faz 2: Ep {PHASE_1_END+1}-{PHASE_2_END} (vs Self-Play)")
    print(f"   Faz 3: Ep {PHASE_2_END+1}-{EPISODES} (Mixed)")
    print()

    for episode in range(1, EPISODES + 1):

        # ─── FAZ SEÇİMİ: Rakipleri belirle ───
        if episode <= PHASE_1_END:
            opponents = heuristic_bots
        elif episode <= PHASE_2_END:
            opponents = [
                SelfPlayBot(selfplay_model, engine),
                SelfPlayBot(selfplay_model, engine),
                SelfPlayBot(selfplay_model, engine),
            ]
        else:
            # Mixed: %50 self-play, %50 heuristic
            if random.random() < 0.5:
                opponents = [SelfPlayBot(selfplay_model, engine)] * 3
            else:
                opponents = heuristic_bots

        engine.reset()

        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  np.exp(-1. * episode / EPSILON_DECAY)

        pending_experience = None
        total_episode_reward = 0

        while not engine.game_over:
            player_idx = engine.turn
            legal_moves = engine.get_legal_moves(player_idx)

            if not legal_moves:
                engine.game_over = True
                break

            # ═════ SIRA AI'DA (Player 0) ═════
            if player_idx == 0:
                # Önceki bekleyen deneyimi tamamla
                if pending_experience:
                    state, action, acc_reward = pending_experience
                    next_state = get_state_vector(engine, 0)
                    next_mask = get_legal_mask(engine, 0, deck_order)
                    memory.push(state, action, acc_reward, next_state, next_mask, 0.0)
                    pending_experience = None

                # Durum vektörü
                state_np = get_state_vector(engine, 0)
                mask_np = get_legal_mask(engine, 0, deck_order)

                # Hamle seçimi (epsilon-greedy)
                if random.random() < epsilon:
                    legal_indices = [i for i, m in enumerate(mask_np) if m == 1.0]
                    action_idx = random.choice(legal_indices)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(device)
                        q_values = policy_net(state_tensor)
                        mask_tensor = torch.FloatTensor(mask_np).to(device)
                        masked_q = q_values + (1.0 - mask_tensor) * -1e9
                        action_idx = torch.argmax(masked_q).item()

                card_to_play = deck_order[action_idx]

                # Önceki skorları kaydet
                prev_ai_score = engine.penalties[0]
                prev_opp_scores = engine.penalties[:]

                # Hamleyi oyna
                trick_len_before = len(engine.current_trick)
                engine.play_move(0, card_to_play)
                trick_resolved = (trick_len_before == 3)  # 4. kart, el çözüldü

                # Ödül hesapla
                reward = compute_reward(
                    engine, 0, card_to_play,
                    prev_ai_score, prev_opp_scores,
                    trick_resolved
                )

                # Kız çakma metrikleri
                if trick_resolved and card_to_play[1] == 12 and engine.penalties[0] == prev_ai_score:
                    queen_dumps += 1

                total_episode_reward += reward
                pending_experience = (state_np, action_idx, reward)

            # ═════ SIRA RAKİPLERDE ═════
            else:
                bot = opponents[player_idx - 1]
                prev_ai_score = engine.penalties[0]
                prev_opp_scores = engine.penalties[:]

                move = bot.choose_move(legal_moves, engine, player_idx)
                engine.play_move(player_idx, move)

                # Rakip oynarken AI ceza yediyse, bekleyen deneyime ekle
                ai_pain = engine.penalties[0] - prev_ai_score
                if ai_pain < 0 and pending_experience:
                    s, a, r = pending_experience
                    pending_experience = (s, a, r + (ai_pain * 3.0))

                # Rakip ceza yediyse AI'a küçük ödül
                for opp in [1, 2, 3]:
                    opp_pain = engine.penalties[opp] - prev_opp_scores[opp]
                    if opp_pain < 0 and pending_experience:
                        s, a, r = pending_experience
                        pending_experience = (s, a, r + 1.0)

        # ─── EPISODE SONU ───
        if pending_experience:
            state, action, acc_reward = pending_experience
            next_state = get_state_vector(engine, 0)
            next_mask = np.zeros(52)
            memory.push(state, action, acc_reward, next_state, next_mask, 1.0)

        # ─── OPTIMIZATION ───
        if len(memory) > BATCH_SIZE:
            states, actions, rewards, next_states, next_masks, dones = memory.sample(BATCH_SIZE)

            curr_q = policy_net(states).gather(1, actions)

            with torch.no_grad():
                # Double DQN: policy seçer, target değerlendirir
                policy_next_q = policy_net(next_states)
                masked_policy_q = policy_next_q + (1.0 - next_masks) * -1e9
                best_actions = masked_policy_q.argmax(1, keepdim=True)

                target_next_q = target_net(next_states)
                max_next_q = target_next_q.gather(1, best_actions)

                target_q = rewards + (1 - dones) * GAMMA * max_next_q

            loss = nn.SmoothL1Loss()(curr_q, target_q)  # Huber loss, MSE'den daha stabil

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

        # Target network güncelleme
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Self-play modelini periyodik güncelle
        if episode % 500 == 0:
            selfplay_model.load_state_dict(policy_net.state_dict())
            selfplay_model.eval()

        # ─── METRİK TAKIP ───
        running_reward += total_episode_reward
        running_ai_penalty += engine.penalties[0]

        if episode % 200 == 0:
            avg_r = running_reward / 200
            avg_p = running_ai_penalty / 200
            phase = "Heuristic" if episode <= PHASE_1_END else ("SelfPlay" if episode <= PHASE_2_END else "Mixed")
            print(f"Ep {episode:>5d} | ε:{epsilon:.3f} | Avg Reward:{avg_r:>7.1f} | "
                  f"Avg Penalty:{avg_p:>6.1f} | Q-Dumps:{queen_dumps:>3d} | Phase:{phase}")
            running_reward = 0.0
            running_ai_penalty = 0.0
            queen_dumps = 0

        # Milestone kayıtları
        if episode % 2000 == 0:
            checkpoint = f"king_ai_master_ep{episode}.pth"
            torch.save(policy_net.state_dict(), checkpoint)
            print(f"   💾 Checkpoint: {checkpoint}")

    # ─── EĞİTİM BİTTİ ───
    torch.save(policy_net.state_dict(), MODEL_SAVE)
    print(f"\n👑 MASTER EĞİTİM TAMAMLANDI! → {MODEL_SAVE}")

    # Veri setini kaydet
    memory.save_dataset("data/master_rl_dataset.npz")


if __name__ == "__main__":
    train_master()
