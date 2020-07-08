from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),

    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight,AKnave))),
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight,AKnave))),
    Implication(BKnight, And(Or(BKnight, BKnave))),
    Implication(BKnave, Not(Or(AKnight,BKnave))),
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    
    Implication(AKnight, Or(And(AKnight, AKnight), And(AKnave, BKnave))),
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    Implication(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))),
    Implication(BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight)))),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
ASaysKnight = Symbol("A Says I am a Knight.")
ASaysKnave = Symbol("A Says I am a Knave.")

knowledge3 = And(
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    And(Or(CKnight, CKnave), Not(And(CKnight, CKnave))),

    And(Or(ASaysKnight, ASaysKnave), Not(And(ASaysKnight, ASaysKnave))),


    Implication(AKnight, ASaysKnight),
    Implication(BKnight, ASaysKnave),
    Implication(BKnight, CKnave),
    Implication(CKnight, AKnight),
   
    Implication(AKnave, ASaysKnight),
    Implication(BKnave, ASaysKnight),
    Implication(BKnave, Not(CKnave)),
    Implication(CKnave, Not(AKnight)),

)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
