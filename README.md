# Chess RL
A self-play reinforcement learning chess agent built from scratch.



## Limitations
- Underpromotion is not supported: a pawn reaching the last rank always promotes to a queen.
- The network's input does not include castling rights, en passant availability, or repetition history. Move legality is handled by python-chess, so all of these moves are still played correctly. The network just can't use them as signals when evaluating a position.