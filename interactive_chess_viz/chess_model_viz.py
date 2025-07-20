import streamlit as st
import chess
import chess.svg
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from chess_utils import board_to_tensor, action_to_move
from model import ChessModel
import os
from typing import Dict, List, Tuple
import base64

st.set_page_config(
    page_title="Chess-SL Model Explorer", 
    page_icon="♛", 
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

def render_chess_board_svg(board: chess.Board, move_probs: torch.Tensor = None, top_k: int = 10) -> str:
    """Render chess board as SVG with move probability arrows"""
    if move_probs is not None:
        # Get top k moves
        top_probs, top_actions = torch.topk(move_probs, min(top_k, len(move_probs)))
        
        arrows = []
        for prob, action in zip(top_probs, top_actions):
            if prob > 0.01:  # Only show moves with >1% probability
                try:
                    move = action_to_move(action.item(), board)
                    if move in board.legal_moves:
                        # Color intensity based on probability
                        alpha = min(255, int(prob * 512))
                        arrow_color = f"#00aa00{alpha:02x}"
                        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=arrow_color))
                except:
                    continue
        
        return chess.svg.board(board, arrows=arrows, size=400)
    else:
        return chess.svg.board(board, size=400)

def create_move_probability_heatmap(move_probs: torch.Tensor, board: chess.Board) -> go.Figure:
    """Create a heatmap of move probabilities"""
    # Reshape to 64x64 (from_square x to_square)
    prob_matrix = move_probs.view(64, 64).numpy()
    
    # Create square labels
    square_names = [chess.square_name(i) for i in range(64)]
    
    fig = go.Figure(data=go.Heatmap(
        z=prob_matrix,
        x=square_names,
        y=square_names,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Move Probability Matrix (From → To)",
        xaxis_title="To Square",
        yaxis_title="From Square",
        width=600,
        height=600
    )
    
    return fig

def create_legal_vs_predicted_chart(board: chess.Board, move_probs: torch.Tensor) -> go.Figure:
    """Compare legal moves vs predicted probabilities"""
    legal_moves = list(board.legal_moves)
    legal_actions = []
    legal_probs = []
    
    for move in legal_moves:
        try:
            action = move.from_square * 64 + move.to_square
            prob = move_probs[action].item()
            legal_actions.append(f"{move}")
            legal_probs.append(prob)
        except:
            continue
    
    # Sort by probability
    sorted_data = sorted(zip(legal_actions, legal_probs), key=lambda x: x[1], reverse=True)
    moves, probs = zip(*sorted_data) if sorted_data else ([], [])
    
    fig = go.Figure(data=go.Bar(
        x=list(moves),
        y=list(probs),
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Predicted Probabilities for Legal Moves",
        xaxis_title="Legal Moves",
        yaxis_title="Predicted Probability",
        xaxis_tickangle=-45
    )
    
    return fig

def main():
    st.title("🏆 Chess-SL Model Explorer")
    st.markdown("Interactive exploration of chess move prediction models")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model selection
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
    
    # Position input
    st.header("Chess Position")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Position")
        
        # Position input methods
        input_method = st.radio("Input Method", ["FEN String", "Starting Position", "Random Position"])
        
        if input_method == "FEN String":
            fen = st.text_input("FEN String", value="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        elif input_method == "Starting Position":
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        else:  # Random Position
            if st.button("Generate Random Position"):
                # Create a random position by playing random moves
                board_temp = chess.Board()
                for _ in range(np.random.randint(5, 20)):
                    if board_temp.is_game_over():
                        break
                    legal_moves = list(board_temp.legal_moves)
                    board_temp.push(np.random.choice(legal_moves))
                fen = board_temp.fen()
                st.session_state.random_fen = fen
            
            fen = getattr(st.session_state, 'random_fen', "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        # Create board from FEN
        try:
            board = chess.Board(fen)
            
            # Display basic board info
            st.write(f"**Turn:** {'White' if board.turn else 'Black'}")
            st.write(f"**Legal Moves:** {len(list(board.legal_moves))}")
            st.write(f"**Castling Rights:** {board.castling_rights}")
            st.write(f"**En Passant:** {board.ep_square if board.ep_square else 'None'}")
            
        except ValueError as e:
            st.error(f"Invalid FEN: {e}")
            st.stop()
    
    with col2:
        st.subheader("Model Analysis")
        
        # Get model predictions
        move_probs = get_model_predictions(model, board)
        
        # Top-k moves to display
        top_k = st.slider("Show Top K Moves", 1, 20, 10)
        
        # Analysis options
        show_heatmap = st.checkbox("Show Probability Heatmap", value=True)
        show_legal_comparison = st.checkbox("Show Legal Moves Analysis", value=True)
    
    # Main visualization area
    st.header("Model Predictions Visualization")
    
    # Chess board with arrows
    board_svg = render_chess_board_svg(board, move_probs, top_k)
    
    # Display SVG board
    st.subheader("Chess Board with Predicted Moves")
    st.markdown(f'<div style="text-align: center">{board_svg}</div>', unsafe_allow_html=True)
    
    # Additional visualizations
    col3, col4 = st.columns([1, 1])
    
    with col3:
        if show_legal_comparison:
            st.subheader("Legal Moves Analysis")
            legal_chart = create_legal_vs_predicted_chart(board, move_probs)
            st.plotly_chart(legal_chart, use_container_width=True)
    
    with col4:
        if show_heatmap:
            st.subheader("Full Probability Heatmap")
            heatmap = create_move_probability_heatmap(move_probs, board)
            st.plotly_chart(heatmap, use_container_width=True)
    
    # Detailed statistics
    st.header("Detailed Statistics")
    
    legal_moves = list(board.legal_moves)
    legal_move_probs = []
    illegal_move_probs = []
    
    for i, prob in enumerate(move_probs):
        try:
            move = action_to_move(i, board)
            if move in legal_moves:
                legal_move_probs.append(prob.item())
            else:
                illegal_move_probs.append(prob.item())
        except:
            illegal_move_probs.append(prob.item())
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Total Probability on Legal Moves", f"{sum(legal_move_probs):.4f}")
    
    with col6:
        st.metric("Total Probability on Illegal Moves", f"{sum(illegal_move_probs):.4f}")
    
    with col7:
        st.metric("Max Legal Move Probability", f"{max(legal_move_probs) if legal_move_probs else 0:.4f}")
    
    # Top predictions table
    st.subheader("Top Predicted Moves")
    
    top_probs, top_actions = torch.topk(move_probs, min(20, len(move_probs)))
    
    move_data = []
    for prob, action in zip(top_probs, top_actions):
        try:
            move = action_to_move(action.item(), board)
            is_legal = move in legal_moves
            move_data.append({
                "Move": str(move),
                "Probability": f"{prob.item():.4f}",
                "Legal": "✅" if is_legal else "❌",
                "From": chess.square_name(move.from_square),
                "To": chess.square_name(move.to_square)
            })
        except:
            continue
    
    if move_data:
        st.dataframe(move_data, use_container_width=True)
    
    # Model comparison section
    st.header("Model Comparison")
    st.markdown("Compare different model configurations on the same position:")
    
    if st.button("Compare with Different Architecture"):
        st.info("Feature coming soon: Compare multiple models side-by-side")

if __name__ == "__main__":
    main()