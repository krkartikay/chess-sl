import streamlit as st
import chess
import chess.pgn
import torch
import numpy as np
import plotly.graph_objects as go
from chess_utils import board_to_tensor, action_to_move
from model import ChessModel
import io
from typing import List, Dict
import os

st.set_page_config(
    page_title="Chess Game Explorer", 
    page_icon="🎮", 
    layout="wide"
)

@st.cache_resource
def load_model(model_path: str, config: Dict) -> ChessModel:
    """Load a trained chess model"""
    model = ChessModel(
        n_blocks=config['n_blocks'],
        n_channels=config['n_channels'],
        n_hidden=config['n_hidden'],
        filter_size=config.get('filter_size', 3)
    )
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
    return model

def get_model_predictions(model: ChessModel, board: chess.Board) -> torch.Tensor:
    """Get model predictions for a chess position"""
    position_tensor = board_to_tensor(board).unsqueeze(0)
    
    if torch.cuda.is_available():
        position_tensor = position_tensor.cuda()
    
    with torch.no_grad():
        predictions = model(position_tensor)
    
    return predictions.squeeze().cpu()

def analyze_game_with_model(pgn_text: str, model: ChessModel) -> List[Dict]:
    """Analyze a PGN game with the model"""
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    
    if game is None:
        return []
    
    analysis = []
    board = game.board()
    
    for i, move in enumerate(game.mainline_moves()):
        # Get model predictions for current position
        move_probs = get_model_predictions(model, board)
        
        # Find probability of the actual move played
        try:
            actual_action = move.from_square * 64 + move.to_square
            actual_prob = move_probs[actual_action].item()
        except:
            actual_prob = 0.0
        
        # Get top predicted move
        legal_moves = list(board.legal_moves)
        legal_probs = []
        legal_actions = []
        
        for legal_move in legal_moves:
            action = legal_move.from_square * 64 + legal_move.to_square
            prob = move_probs[action].item()
            legal_probs.append(prob)
            legal_actions.append(legal_move)
        
        if legal_probs:
            best_idx = np.argmax(legal_probs)
            best_move = legal_actions[best_idx]
            best_prob = legal_probs[best_idx]
        else:
            best_move = None
            best_prob = 0.0
        
        # Calculate rank of actual move
        move_rank = 1
        for prob in legal_probs:
            if prob > actual_prob:
                move_rank += 1
        
        analysis.append({
            'move_number': (i // 2) + 1,
            'color': 'White' if i % 2 == 0 else 'Black',
            'actual_move': str(move),
            'actual_prob': actual_prob,
            'best_move': str(best_move) if best_move else "None",
            'best_prob': best_prob,
            'move_rank': move_rank,
            'total_legal_moves': len(legal_moves),
            'fen': board.fen()
        })
        
        board.push(move)
    
    return analysis

def create_accuracy_plot(analysis: List[Dict]) -> go.Figure:
    """Create accuracy plot over moves"""
    move_numbers = [item['move_number'] for item in analysis]
    actual_probs = [item['actual_prob'] for item in analysis]
    colors = ['blue' if item['color'] == 'White' else 'red' for item in analysis]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=move_numbers,
        y=actual_probs,
        mode='markers+lines',
        marker=dict(color=colors),
        name='Actual Move Probability',
        hovertemplate='Move %{x}: %{customdata}<br>Probability: %{y:.4f}<extra></extra>',
        customdata=[item['actual_move'] for item in analysis]
    ))
    
    fig.update_layout(
        title="Model's Probability for Actual Moves Played",
        xaxis_title="Move Number",
        yaxis_title="Predicted Probability",
        showlegend=True
    )
    
    return fig

def create_ranking_plot(analysis: List[Dict]) -> go.Figure:
    """Create move ranking plot"""
    move_numbers = [item['move_number'] for item in analysis]
    move_ranks = [item['move_rank'] for item in analysis]
    colors = ['blue' if item['color'] == 'White' else 'red' for item in analysis]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=move_numbers,
        y=move_ranks,
        mode='markers+lines',
        marker=dict(color=colors),
        name='Move Rank',
        hovertemplate='Move %{x}: %{customdata}<br>Rank: %{y}<extra></extra>',
        customdata=[item['actual_move'] for item in analysis]
    ))
    
    fig.update_layout(
        title="Rank of Actual Move Among Model Predictions",
        xaxis_title="Move Number",
        yaxis_title="Move Rank (1 = Best)",
        showlegend=True,
        yaxis=dict(autorange='reversed')  # Lower rank (better) at top
    )
    
    return fig

def main():
    st.title("🎮 Chess Game Explorer")
    st.markdown("Analyze chess games with your trained models")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model selection (same as in main viz)
    model_files = []
    for root, dirs, files in os.walk("../"):
        for file in files:
            if file.endswith('.pth') and 'model' in file:
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        st.error("No model files found! Train a model first.")
        st.stop()
    
    selected_model = st.sidebar.selectbox("Select Model", model_files)
    
    # Model architecture configuration
    st.sidebar.subheader("Model Architecture")
    n_blocks = st.sidebar.slider("Number of Conv Blocks", 0, 8, 8)
    n_channels = st.sidebar.slider("Number of Channels", 4, 128, 128)
    n_hidden = st.sidebar.selectbox("Hidden Layer Size", [1024, 4096], index=1)
    
    config = {
        'n_blocks': n_blocks,
        'n_channels': n_channels,
        'n_hidden': n_hidden,
        'filter_size': 3
    }
    
    # Load model
    try:
        model = load_model(selected_model, config)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Game input
    st.header("Game Input")
    
    input_method = st.radio("Input Method", ["PGN Text", "Famous Games", "Upload PGN File"])
    
    if input_method == "PGN Text":
        pgn_text = st.text_area("Paste PGN here:", height=200, 
                               value="""[Event "Example Game"]
[Site "Chess.com"]
[Date "2024.01.01"]
[White "Player 1"]
[Black "Player 2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7""")
    
    elif input_method == "Famous Games":
        famous_games = {
            "Morphy vs. Duke of Brunswick (1858)": """[Event "Opera Game"]
[Site "Paris"]
[Date "1858.10.21"]
[White "Paul Morphy"]
[Black "Duke of Brunswick"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0""",
            
            "Kasparov vs. Topalov (1999)": """[Event "Hoogovens A Tournament"]
[Site "Wijk aan Zee NED"]
[Date "1999.01.20"]
[White "Garry Kasparov"]
[Black "Veselin Topalov"]
[Result "1-0"]

1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. Be3 Bg7 5. Qd2 c6 6. f3 b5 7. Nge2 Nbd7 8. Bh6 Bxh6 9. Qxh6 Bb7 10. a3 e5 11. O-O-O Qe7 12. Kb1 a6 13. Nc1 O-O-O 14. Nb3 exd4 15. Rxd4 c5 16. Rd1 Nb6 17. g3 Kb8 18. Na5 Ba8 19. Bh3 d5 20. Qf4+ Ka7 21. Rhe1 d4 22. Nd5 Nbxd5 23. exd5 Qd6 24. Rxd4 cxd4 25. Re7+ Kb6 26. Qxd4+ Kxa5 27. b4+ Ka4 28. Qc3 Qxd5 29. Ra7 Bb7 30. Rxb7 Qc4 31. Qxf6 Kxa3 32. Qxa6+ Kxb4 33. c3+ Kxc3 34. Qa1+ Kd2 35. Qb2+ Kd1 36. Bf1 Rd2 37. Rd7 Rxd7 38. Bxc4 bxc4 39. Qxh8 Rd3 40. Qa8 c3 41. Qa4+ Ke1 42. f4 f5 43. Kc1 Rd2 44. Qa7 1-0"""
        }
        
        selected_game = st.selectbox("Select Famous Game", list(famous_games.keys()))
        pgn_text = famous_games[selected_game]
        
    else:  # Upload PGN File
        uploaded_file = st.file_uploader("Choose a PGN file", type="pgn")
        if uploaded_file is not None:
            pgn_text = str(uploaded_file.read(), "utf-8")
        else:
            pgn_text = ""
    
    if st.button("Analyze Game") and pgn_text:
        with st.spinner("Analyzing game..."):
            analysis = analyze_game_with_model(pgn_text, model)
        
        if not analysis:
            st.error("Could not parse the PGN. Please check the format.")
            st.stop()
        
        st.success(f"✅ Analyzed {len(analysis)} moves!")
        
        # Display results
        st.header("Game Analysis Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_prob = np.mean([item['actual_prob'] for item in analysis])
            st.metric("Average Move Probability", f"{avg_prob:.4f}")
        
        with col2:
            top1_accuracy = sum(1 for item in analysis if item['move_rank'] == 1) / len(analysis)
            st.metric("Top-1 Accuracy", f"{top1_accuracy:.2%}")
        
        with col3:
            top3_accuracy = sum(1 for item in analysis if item['move_rank'] <= 3) / len(analysis)
            st.metric("Top-3 Accuracy", f"{top3_accuracy:.2%}")
        
        with col4:
            avg_rank = np.mean([item['move_rank'] for item in analysis])
            st.metric("Average Move Rank", f"{avg_rank:.1f}")
        
        # Plots
        col5, col6 = st.columns(2)
        
        with col5:
            accuracy_plot = create_accuracy_plot(analysis)
            st.plotly_chart(accuracy_plot, use_container_width=True)
        
        with col6:
            ranking_plot = create_ranking_plot(analysis)
            st.plotly_chart(ranking_plot, use_container_width=True)
        
        # Detailed analysis table
        st.subheader("Move-by-Move Analysis")
        
        # Add filters
        show_only_mistakes = st.checkbox("Show only moves ranked >5")
        min_prob = st.slider("Minimum probability threshold", 0.0, 1.0, 0.0, 0.01)
        
        filtered_analysis = analysis
        if show_only_mistakes:
            filtered_analysis = [item for item in analysis if item['move_rank'] > 5]
        if min_prob > 0:
            filtered_analysis = [item for item in filtered_analysis if item['actual_prob'] >= min_prob]
        
        # Create display dataframe
        display_data = []
        for item in filtered_analysis:
            display_data.append({
                "Move #": item['move_number'],
                "Color": item['color'],
                "Actual Move": item['actual_move'],
                "Prob.": f"{item['actual_prob']:.4f}",
                "Rank": item['move_rank'],
                "Model's Best": item['best_move'],
                "Best Prob.": f"{item['best_prob']:.4f}",
                "Legal Moves": item['total_legal_moves']
            })
        
        if display_data:
            st.dataframe(display_data, use_container_width=True)
        else:
            st.info("No moves match the current filters.")
        
        # Position explorer for specific moves
        st.subheader("Position Explorer")
        
        move_idx = st.selectbox(
            "Select move to explore:",
            range(len(analysis)),
            format_func=lambda x: f"Move {analysis[x]['move_number']} ({analysis[x]['color']}): {analysis[x]['actual_move']}"
        )
        
        if move_idx is not None:
            selected_item = analysis[move_idx]
            board = chess.Board(selected_item['fen'])
            
            col7, col8 = st.columns([1, 1])
            
            with col7:
                # Display board
                board_svg = chess.svg.board(board, size=300)
                st.markdown(f'<div style="text-align: center">{board_svg}</div>', unsafe_allow_html=True)
            
            with col8:
                st.write(f"**Position after move {selected_item['move_number']}**")
                st.write(f"**Turn:** {selected_item['color']}")
                st.write(f"**Actual move:** {selected_item['actual_move']}")
                st.write(f"**Model probability:** {selected_item['actual_prob']:.4f}")
                st.write(f"**Move rank:** {selected_item['move_rank']} / {selected_item['total_legal_moves']}")
                st.write(f"**Model's best:** {selected_item['best_move']} ({selected_item['best_prob']:.4f})")

if __name__ == "__main__":
    main()