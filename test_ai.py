import torch
import random
import numpy as np
from engine.game_engine import KizAlmazEngine
from ai.model import KingDQN, get_state_vector

def test_model(num_games=100):
    print(f"--- TEST BAŞLIYOR: Akıllı Agent vs 3 Rastgele Bot ({num_games} Oyun) ---")
    
    engine = KizAlmazEngine()
    
    # 1. Eğitilmiş Beyni Yükle
    model = KingDQN()
    try:
        model.load_state_dict(torch.load("king_ai_master.pth", weights_only=True))
        model.eval() # Modeli 'Test Modu'na al (Öğrenme kapalı)
        print("Eğitilmiş model başarıyla yüklendi!")
    except FileNotFoundError:
        print("Model dosyası bulunamadı! Önce train_cycle.py çalıştırın.")
        return

    ai_total_score = 0
    random_bots_total_score = 0
    deck_order = [(s, r) for s in engine.suits for r in engine.ranks]

    for i in range(num_games):
        engine.reset()
        
        while not engine.game_over:
            player_idx = engine.turn
            legal_moves = engine.get_legal_moves(player_idx)
            
            if not legal_moves:
                engine.game_over = True
                break

            if player_idx == 0:
                # --- BİZİM EĞİTİLMİŞ AI (Player 0) ---
                state_np = get_state_vector(engine, player_idx)
                state_tensor = torch.FloatTensor(state_np)
                
                with torch.no_grad():
                    q_values = model(state_tensor)
                    # Maskeleme (Sadece legal hamleler)
                    mask = torch.full(q_values.shape, float('-inf'))
                    legal_indices = [deck_order.index(m) for m in legal_moves]
                    for idx in legal_indices:
                        mask[idx] = q_values[idx]
                    action_idx = torch.argmax(mask).item()
                
                card_to_play = deck_order[action_idx]
            else:
                # --- RASTGELE BOTLAR (Player 1, 2, 3) ---
                card_to_play = random.choice(legal_moves)
            
            engine.play_move(player_idx, card_to_play)
        
        # Oyun bitti, skorları topla
        ai_score = engine.penalties[0]
        others_score = sum(engine.penalties[1:])
        
        ai_total_score += ai_score
        random_bots_total_score += others_score

    # SONUÇ RAPORU
    avg_ai = ai_total_score / num_games
    avg_random = random_bots_total_score / (num_games * 3) # 3 bot var
    
    print("\n" + "="*40)
    print(f"SONUÇLAR ({num_games} Oyun Sonunda):")
    print(f"AI Ortalama Ceza: {avg_ai:.2f}")
    print(f"Rastgele Bot Ortalama : {avg_random:.2f}")
    print("="*40)
    
    if avg_ai > avg_random: # Dikkat: Skorlar negatif, yani -5 > -20 (Daha iyi)
        print("Model rastgele oynamaktan daha iyi!")
    else:
        print("Model hala rastgele gibi oynuyor")

if __name__ == "__main__":
    test_model()