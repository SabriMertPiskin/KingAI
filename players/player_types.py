import random
import torch
from ai.model import KingDQN, get_state_vector

class Player:
    def __init__(self, name):
        self.name = name

    def choose_move(self, legal_moves, table):
        raise NotImplementedError

class ManualPlayer(Player):
    def choose_move(self, legal_moves, table):
        print(f"\n>> {self.name}'in Sırası. Yerdeki Kağıtlar: {table}")
        for i, card in enumerate(legal_moves):
            print(f"{i}: {card[1]} of {card[0]}")
        
        while True:
            try:
                choice = int(input("Seçiminiz (indis): "))
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice]
            except: pass
            print("Hatalı giriş!")

class BotPlayer(Player):
    def choose_move(self, legal_moves, table, player_idx=None):
        # Şimdilik bencil olmayan, rastgele bot
        return random.choice(legal_moves)

class SmartBotPlayer(Player):
    def __init__(self, name, model_path="king_ai_ultimate.pth"):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KingDQN().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            print(f"{name} akıllı beyni yüklendi! ({model_path})")
        except FileNotFoundError:
            print(f"Uyarı: {model_path} bulunamadı! Rastgele oynayacak.")
            self.model = None

    def choose_move(self, legal_moves, engine, player_idx):
        if not self.model:
            return random.choice(legal_moves)
            
        state_np = get_state_vector(engine, player_idx)
        state_tensor = torch.FloatTensor(state_np).to(self.device)
        
        deck_order = [(s, r) for s in engine.suits for r in engine.ranks]
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            
            # Masking out illegal moves
            mask = torch.full(q_values.shape, float('-inf')).to(self.device)
            legal_indices = [deck_order.index(m) for m in legal_moves]
            for idx in legal_indices:
                mask[idx] = q_values[idx]
                
            action_idx = torch.argmax(mask).item()
            
        return deck_order[action_idx]