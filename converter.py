# converter.py

# Fixed color mappings
pattern_color_map = {
    'U': 'sarı',
    'D': 'beyaz',
    'F': 'mavi',
    'B': 'yeşil',
    'L': 'turuncu',
    'R': 'kırmızı',
}

target_color_map = {
    'U': 'sarı',
    'D': 'beyaz',
    'F': 'kırmızı',
    'B': 'turuncu',
    'L': 'mavi',
    'R': 'yeşil',
}


def formatSolution(sequence):
    pattern_color_to_face = {v: k for k, v in pattern_color_map.items()}
    target_color_to_face = {v: k for k, v in target_color_map.items()}

    # Map each face based on shared color meaning
    face_map = {
        pattern_color_to_face[color]: target_color_to_face[color]
        for color in pattern_color_to_face
    }

    converted_moves = []
    tokens = sequence.strip().split()
    for move in tokens:
        face = move[0]
        suffix = move[1:] if len(move) > 1 else ''
        if face not in face_map:
            continue  # or raise an error
        new_face = face_map[face]
        if suffix == "'":
            num = "3"
        elif suffix == "2":
            num = "2"
        else:
            num = "1"
        converted_moves.append(f"{new_face}{num}")

    return ' '.join(converted_moves)

print(formatSolution("U' R2 L2 F2 B2 U' R L F B' U F2 D2 R2 L2 F2 U2 F2 U' F2"))
