import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import os
from engine.game_engine import KizAlmazEngine
from ai.model import KingDQN, get_state_vector
from players.heuristic_bot import HeuristicBot
from ai.memory import ReplayBuffer

# --- SIKI YÖNETİM AYARLARI ---
EPISODES = 5000          # 5000 oyun yeterli
BATCH_SIZE = 128         # Batch size'ı büyüttük, daha genel baksın
GAMMA = 0.99             
EPSILON_START = 1.0
EPSILON_END = 0.02       # %2'ye kadar düşsün
EPSILON_DECAY = 3000     
TARGET_UPDATE = 50       # Hedef ağı daha sık güncelle
LR = 0.0002              # Learning Rate
MEMORY_SIZE = 100000     # Hafıza geniş

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_legal_mask(engine, player_idx, deck_order):
    """Verilen durum için yasal hamle maskesi oluşturur (1=Legal, 0=Illegal)"""
    legal_moves = engine.get_legal_moves(player_idx)
    mask = np.zeros(52, dtype=np.float32)
    if not legal_moves: return mask 
    
    for move in legal_moves:
        idx = deck_order.index(move)
        mask[idx] = 1.0
    return mask

def train_ultimate():
    engine = KizAlmazEngine()
    
    policy_net = KingDQN().to(device)
    target_net = KingDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE, device)
    
    # Rakipler: Kurallı Botlar
    opponents = [HeuristicBot(), HeuristicBot(), HeuristicBot()]
    
    deck_order = [(s, r) for s in engine.suits for r in engine.ranks]
    
    print(f"🔥 ULTIMATE EĞİTİM BAŞLIYOR ({device})...")

    for episode in range(1, EPISODES + 1):
        engine.reset()
        
        # Bekleyen Deneyim (State, Action, Reward)
        pending_experience = None 
        
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  np.exp(-1. * episode / EPSILON_DECAY)

        total_ai_reward = 0

        while not engine.game_over:
            player_idx = engine.turn
            
            legal_moves = engine.get_legal_moves(player_idx)
            if not legal_moves:
                engine.game_over = True
                break

            # ---------------------------------------------
            # SIRA BİZİM AI'DA (Player 0)
            # ---------------------------------------------
            if player_idx == 0:
                if pending_experience:
                    state, action, accumulated_reward = pending_experience
                    
                    next_state = get_state_vector(engine, player_idx)
                    next_mask = get_legal_mask(engine, player_idx, deck_order)
                    
                    memory.push(state, action, accumulated_reward, next_state, next_mask, 0.0)
                    pending_experience = None

                state_np = get_state_vector(engine, player_idx)
                mask_np = get_legal_mask(engine, player_idx, deck_order)
                
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
                
                prev_score = engine.penalties[player_idx]
                engine.play_move(player_idx, card_to_play)
                current_score = engine.penalties[player_idx]
                
                reward = current_score - prev_score
                if reward < 0: reward = -10.0 
                else: 
                    if card_to_play[1] >= 12: reward = 0.5 
                    else: reward = 0.1

                total_ai_reward += reward
                pending_experience = (state_np, action_idx, reward)

            # ---------------------------------------------
            # SIRA RAKİPLERDE
            # ---------------------------------------------
            else:
                bot = opponents[player_idx - 1]
                move = bot.choose_move(legal_moves, engine, player_idx)
                
                prev_ai_score = engine.penalties[0]
                engine.play_move(player_idx, move)
                curr_ai_score = engine.penalties[0]
                
                ai_pain = curr_ai_score - prev_ai_score
                if ai_pain < 0 and pending_experience:
                    s, a, r = pending_experience
                    pending_experience = (s, a, r + (ai_pain * 2.0))

        # --- OYUN SONU (EPISODE BİTTİ) ---
        if pending_experience:
            state, action, accumulated_reward = pending_experience
            next_state = get_state_vector(engine, 0)
            next_mask = np.zeros(52) 
            memory.push(state, action, accumulated_reward, next_state, next_mask, 1.0)


        # --- EĞİTİM (OPTIMIZATION) ---
        if len(memory) > BATCH_SIZE:
            states, actions, rewards, next_states, next_masks, dones = memory.sample(BATCH_SIZE)
            
            curr_q = policy_net(states).gather(1, actions)
            
            with torch.no_grad():
                next_q_raw = target_net(next_states)
                masked_next_q = next_q_raw + (1.0 - next_masks) * -1e9
                max_next_q = masked_next_q.max(1)[0].unsqueeze(1)
                target_q = rewards + (1 - dones) * GAMMA * max_next_q
            
            loss = nn.MSELoss()(curr_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0) 
            optimizer.step()

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 1. MİLESTONE LOGLAMA (Her 500 oyunda bir JSON kaydet)
        if episode % 500 == 0:
            try:
                saved_log = engine.save_full_log()
                yeni_ad = f"data/game_log_episode_{episode}.json"
                os.rename(saved_log, yeni_ad)
            except FileNotFoundError:
                pass

        if episode % 100 == 0:
            avg_pen = engine.penalties[0]
            print(f"Ep: {episode}, Eps: {epsilon:.2f}, Last Reward: {total_ai_reward:.1f}, AI Score: {avg_pen}")

    # --- EĞİTİM BİTTİ, MODELLERİ VE VERİYİ KAYDET ---
    torch.save(policy_net.state_dict(), "king_ai_ultimate.pth")
    print("👑 ULTIMATE Eğitim Bitti! Model ağırlıkları kaydedildi.")
    
    # 3. VERİ SETİNİ KAYDET (The Dataset)
    memory.save_dataset("data/king_rl_dataset.npz")

if __name__ == "__main__":
    train_ultimate()