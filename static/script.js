const suitSymbols = { 'Spade': '♠', 'Heart': '♥', 'Diamond': '♦', 'Club': '♣' };
const rankMap = { 11: 'J', 12: 'Q', 13: 'K', 14: 'A' };

// DOM Elements
const btnStart = document.getElementById('btn-start');
const handDiv = document.getElementById('my-hand');
const trickArea = document.getElementById('trick-area');
const toast = document.getElementById('toast');
const gameState = { isOurTurn: false, game_over: false, intervalId: null };
let lastActionsCount = -1;

function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

function formatRank(rank) {
    return rankMap[rank] || rank;
}

function createCardHTML(suit, rank, extraClasses = '') {
    const symbol = suitSymbols[suit];
    const displayRank = formatRank(rank);
    return `
        <div class="card suit-${suit} ${extraClasses}" data-suit="${suit}" data-rank="${rank}">
            <div class="rank-top">${displayRank} ${symbol}</div>
            <div class="suit-center">${symbol}</div>
        </div>
    `;
}

async function fetchState() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();
        updateUI(data);
    } catch(err) {
        console.error("Failed to fetch state:", err);
    }
}

async function startGame() {
    showToast("Starting new game with Ultimate AIs...");
    try {
        const res = await fetch('/api/start', { method: 'POST' });
        const data = await res.json();
        document.getElementById('game-over-modal').classList.add('hidden');
        if(gameState.intervalId) clearInterval(gameState.intervalId);
        updateUI(data);
    } catch(err) {
        showToast("Error starting game.");
    }
}

async function playCard(suit, rank) {
    if(!gameState.isOurTurn) return;
    try {
        const res = await fetch('/api/play', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ suit, rank })
        });
        if(!res.ok) {
            showToast("Illegal Move!");
            return;
        }
        const data = await res.json();
        updateUI(data);
    } catch(err) {
        console.error(err);
    }
}

// Bot step loop
async function processBotStep() {
    if (gameState.isOurTurn || gameState.game_over) return;
    try {
        const res = await fetch('/api/step', { method: 'POST' });
        const data = await res.json();
        updateUI(data);
    } catch(err) {
        console.error(err);
    }
}

function updateUI(data) {
    if(!data.started) return;

    gameState.game_over = data.game_over;
    gameState.isOurTurn = (data.turn === 0);

    // Update Scores and Counts
    document.getElementById('my-penalty').textContent = data.penalties[0];
    for (let i = 1; i <= 3; i++) {
        document.querySelector(`#opp-${i} .penalty`).textContent = data.penalties[i];
        document.querySelector(`#opp-${i} .card-count`).textContent = data.hands[i];
    }

    // Render Hand
    handDiv.innerHTML = '';
    const myCards = data.hands[0];
    const legalMoves = data.legal_moves || [];
    
    myCards.sort((a,b) => {
        if(a[0] < b[0]) return -1;
        if(a[0] > b[0]) return 1;
        return a[1] - b[1];
    });

    myCards.forEach(card => {
        const suit = card[0];
        const rank = card[1];
        const isLegal = legalMoves.some(m => m[0] === suit && m[1] === rank);
        
        let extraClasses = '';
        if (gameState.isOurTurn && !isLegal) extraClasses = 'illegal';
        
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = createCardHTML(suit, rank, extraClasses);
        const cardElem = tempDiv.firstElementChild;
        
        if(gameState.isOurTurn && isLegal) {
            cardElem.addEventListener('click', () => playCard(suit, rank));
        }
        handDiv.appendChild(cardElem);
    });

    // Handle Trick Pacing
    const isNewAction = data.actions_count > lastActionsCount;
    lastActionsCount = data.actions_count;

    if (data.current_trick.length === 0 && data.previous_trick && isNewAction) {
        // A trick just finished. Render it momentarily.
        renderTrickCards(data.previous_trick.cards);
        
        // Announce winner
        let winnerName = data.previous_trick.winner === 0 ? "You" : `Agent ${data.previous_trick.winner}`;
        let pts = data.previous_trick.points;
        let msg = `${winnerName} took the trick.`;
        if (pts < 0) {
            msg = `🚨 ${winnerName} took a penalty! (${pts} pts)`;
        }
        showToast(msg);
        
        // Pause 3 seconds then continue
        setTimeout(() => {
            trickArea.innerHTML = ''; 
            checkGameProgression(data);
        }, 3000);
    } else {
        renderTrickCards(data.current_trick);
        checkGameProgression(data);
    }
}

/**
 * Returns the fixed pixel position {left, top} for a player's card slot,
 * computed from the ACTUAL rendered bounding rect of the game-board.
 * trick-area is position:fixed covering the whole viewport, so these
 * viewport-relative pixel coords map directly to trick-area coords.
 */
function getCardSlot(playerIdx) {
    const board = document.querySelector('.game-board');
    const rect  = board.getBoundingClientRect();
    const cx = rect.left + rect.width  / 2;   // horizontal center of board
    const cy = rect.top  + rect.height / 2;   // vertical   center of board
    const h    = 220;   // horizontal offset (Agent 1 left, Agent 3 right)
    const vUp  = 120;   // vertical offset upward (Agent 2 — avoid overlapping label)
    const vDown = 200;  // vertical offset downward (You — closer to hand bar)
    return [
        { left: cx,     top: cy + vDown },   // Player 0 – bottom (You)
        { left: cx - h, top: cy         },   // Player 1 – left   (Agent 1)
        { left: cx,     top: cy - vUp   },   // Player 2 – top    (Agent 2)
        { left: cx + h, top: cy         },   // Player 3 – right  (Agent 3)
    ][playerIdx];
}

function renderTrickCards(trick) {
    trickArea.innerHTML = '';
    trick.forEach((play, index) => {
        const [p_idx, suit, rank] = play;
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = createCardHTML(suit, rank, 'card-played');
        const cardElem = tempDiv.firstElementChild;
        
        // Compute exact pixel slot and apply as inline style
        const pos = getCardSlot(p_idx);
        cardElem.style.left      = pos.left + 'px';
        cardElem.style.top       = pos.top  + 'px';
        cardElem.style.transform = 'translate(-50%, -50%)';
        cardElem.style.zIndex    = index + 10;
        
        trickArea.appendChild(cardElem);
    });
}

function checkGameProgression(data) {
    if (data.game_over) {
        if(gameState.intervalId) clearInterval(gameState.intervalId);
        const modal = document.getElementById('game-over-modal');
        modal.classList.remove('hidden');
        
        let scoresHTML = '';
        data.penalties.forEach((score, idx) => {
            let name = idx === 0 ? "You (Sabri)" : `Agent ${idx}`;
            scoresHTML += `<p>${name}: <span style="font-weight:bold; color: ${score < 0 ? '#ef4444' : '#10b981'}">${score}</span></p>`;
        });
        document.getElementById('final-scores').innerHTML = scoresHTML;
        return;
    }

    if (!gameState.isOurTurn) {
        // Human played a card, now bots take over. We use short delay.
        setTimeout(processBotStep, 600);
    }
}

// Event Listeners
btnStart.addEventListener('click', startGame);
document.getElementById('btn-restart').addEventListener('click', startGame);

// Init
fetchState();
