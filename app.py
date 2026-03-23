import os
from flask import Flask, jsonify, request, render_template

from engine.game_engine import KizAlmazEngine
from players.player_types import ManualPlayer, SmartBotPlayer

app = Flask(__name__)

# Global state for a single-session game. Valid because we run locally.
game_state = {
    'engine': None,
    'players': []
}

@app.route('/')
def index():
    return render_template('index.html')

def get_current_state():
    engine = game_state['engine']
    if not engine:
        return {'game_over': True, 'started': False}
    
    previous_trick = None
    if len(engine.current_trick) == 0 and engine.game_log["tricks"]:
        last_t = engine.game_log["tricks"][-1]
        if len(engine.game_log["actions"]) >= 4:
            last_4 = engine.game_log["actions"][-4:]
            previous_trick = {
                'cards': [(a["player"], a["played_card"][0], a["played_card"][1]) for a in last_4],
                'winner': last_t['winner'],
                'points': last_t['points']
            }
    
    return {
        'started': True,
        'current_trick': [(p_idx, card[0], card[1]) for p_idx, card in engine.current_trick],
        'previous_trick': previous_trick,
        'actions_count': len(engine.game_log.get("actions", [])),
        'hands': {
            0: [(c[0], c[1]) for c in engine.hands[0]],
            1: len(engine.hands[1]),
            2: len(engine.hands[2]),
            3: len(engine.hands[3])
        },
        'penalties': engine.penalties,
        'turn': engine.turn,
        'game_over': engine.game_over,
        'legal_moves': [(m[0], m[1]) for m in engine.get_legal_moves(0)] if engine.turn == 0 else []
    }

@app.route('/api/state', methods=['GET'])
def state():
    return jsonify(get_current_state())

@app.route('/api/start', methods=['POST'])
def start_game():
    engine = KizAlmazEngine()
    players = [
        ManualPlayer("Sabri"),
        SmartBotPlayer("Agent_1", "king_ai_master.pth"),
        SmartBotPlayer("Agent_2", "king_ai_master.pth"),
        SmartBotPlayer("Agent_3", "king_ai_master.pth")
    ]
    game_state['engine'] = engine
    game_state['players'] = players
    engine.reset()
    return jsonify(get_current_state())

@app.route('/api/play', methods=['POST'])
def play_card():
    engine = game_state['engine']
    if not engine or engine.game_over or engine.turn != 0:
        return jsonify({'error': 'Not your turn or game over'}), 400
        
    data = request.json
    card_suit = data.get('suit')
    card_rank = int(data.get('rank'))
    
    legal_moves = engine.get_legal_moves(0)
    human_card = (card_suit, card_rank)
    
    if human_card in legal_moves:
        engine.play_move(0, human_card)
        return jsonify(get_current_state())
    else:
        return jsonify({'error': 'Illegal move'}), 400

@app.route('/api/step', methods=['POST'])
def step():
    engine = game_state['engine']
    players = game_state['players']
    
    if not engine or engine.game_over:
        return jsonify(get_current_state())
        
    curr_idx = engine.turn
    if curr_idx != 0:
        moves = engine.get_legal_moves(curr_idx)
        player = players[curr_idx]
        if hasattr(player, 'choose_move'):
            try:
                move = player.choose_move(moves, engine, curr_idx)
            except TypeError:
                move = player.choose_move(moves, engine.current_trick)
        else:
            import random
            move = random.choice(moves)
            
        engine.play_move(curr_idx, move)

    return jsonify(get_current_state())

if __name__ == '__main__':
    app.run(debug=True, port=5000)
