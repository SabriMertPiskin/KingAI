import random
from collections import deque
import torch
import numpy as np
import os

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, next_mask, done):
        """
        state: Mevcut durum vektörü
        action: Seçilen hamle indexi
        reward: Alınan toplam ödül (sıra tekrar bana gelene kadar)
        next_state: Sıra tekrar bana geldiğindeki durum
        next_mask: Sıra tekrar bana geldiğinde yapabileceğim yasal hamleler (0/1 maskesi)
        done: Oyun bitti mi?
        """
        self.buffer.append((state, action, reward, next_state, next_mask, done))

    def sample(self, batch_size):
        state, action, reward, next_state, next_mask, done = zip(*random.sample(self.buffer, batch_size))

        return (
            torch.FloatTensor(np.array(state)).to(self.device),
            torch.LongTensor(action).unsqueeze(1).to(self.device), 
            torch.FloatTensor(reward).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(next_state)).to(self.device),
            torch.FloatTensor(np.array(next_mask)).to(self.device), 
            torch.FloatTensor(done).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)

    def save_dataset(self, filename="data/rl_experience_dataset.npz"):
        """Hafızadaki tüm deneyimleri sıkıştırılmış Numpy dosyası olarak kaydeder."""
        os.makedirs("data", exist_ok=True)
        
        # Zip'i açarak listeleri ayır
        states, actions, rewards, next_states, next_masks, dones = zip(*self.buffer)
        
        # Numpy formatında sıkıştırarak kaydet
        np.savez_compressed(
            filename,
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.int32),
            rewards=np.array(rewards, dtype=np.float32),
            next_states=np.array(next_states, dtype=np.float32),
            next_masks=np.array(next_masks, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32)
        )
        print(f"💾 Veri seti başarıyla kaydedildi: {filename} ({len(self.buffer)} hamle)")