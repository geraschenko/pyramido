#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#define DEPTH 15
#define HUMAN 0  // are you gunna be there
#define OUTFILE "out.txt"  // where do I record the game
#define VERBOSE 2  //   0 = only output winner
// 1 = "normal" output
// 2 = info about all evaluated moves
#define FIRST LT  // who plays first

#define ILL ((Square)2)
#define EMP ((Square)0)
#define DK ((Square)1)  // Player ``Dark'' is X
#define LT ((Square)-1)  // Player ``Light'' is O
#define SW(x) (-x)
#define CCONV(x) ((1 - x) / 2)  // 0 for DK and 1 for LT

#define HT 5
#define WD 8
#define MAXPIECE 13
#define WINCOUNT 2

#define WIN 10000
#define LOSE (-WIN)

#define C2(a) (((a) * (a) - (a)) / 2)
#define SQCNT (C2(WD + 1) - C2(WD - HT + 1))  // number of board squares

double savequality, prospects;

typedef char Square;

int piececount[2];
int savei, savej, savelr, savevalue;
FILE *outpoint, *weightpoint;
int movenumber = 0;
int moverecord[100][4];

void printsquare(Square x) {
	if (x == EMP) {
		printf(" ");
	} else if (x == DK) {
		printf("X");
	} else if (x == LT) {
		printf("O");
	} else {
		printf("?");
	}
}

void printboard(const Square &current, const Square (&board)[HT][WD]) {
	int i, j;

	printf("\n  ");
	for (i = 0; i < HT - 1; i++) printf("  ");
	for (i = 0; i < WD - HT + 1; i++) printf("+---");
	printf("+\n");
	for (i = 0; i < HT; i++) {
		printf("%d", i);
		for (j = 0; j < HT - i - 1; j++) printf("  ");
		for (j = 0; j < 1 + i + WD - HT; j++) {
			printf(" | ");
			printsquare(board[i][j]);
		}
		printf(" |\n");
		for (j = 0; j < HT - i - 1; j++) printf("  ");
		if (i != HT - 1)
			for (j = 0; j < 2 + i + WD - HT; j++) printf("+-+-");
		else {
			printf("  ");
			for (j = 0; j < 1 + i + WD - HT; j++) printf("+---");
		}
		printf("+\n");
	}
	printf(" ");
	for (j = 0; j < WD; j++) printf("   %d", j);
	printf("\nPieces used: X=%d     O=%d", piececount[0], piececount[1]);
	printf("\nCurrent Player = ");
	printsquare(current);
	printf("\n");
}

void initboard(Square &current, Square (&board)[HT][WD]) {
	int i, j;
	current = FIRST;
	piececount[0] = 0;
	piececount[1] = 0;
	for (i = 0; i < HT; i++) {
		for (j = 0; j < WD; j++) { board[i][j] = (j + HT > i + WD) ? ILL : EMP; }
	}
}

// must current player fall this move?
// returns 1 if have to fall from top, 2 if have to fall elsewhere, 0 otherwise
int fall(const Square &current, const Square (&board)[HT][WD]) {
	int i, j;

	// check top row first
	for (j = 0; j < WD - HT + 1; j++) {
		if ((board[0][j] == current) && (board[1][j] == EMP) &&
		    (board[1][j + 1] == EMP)) {
			return 1;
		}
	}
	for (i = HT - 2; i > 0; i--) {  // skip bottom row
		for (j = 0; j < WD - HT + 1 + i; j++) {
			if ((board[i][j] == current) && (board[i + 1][j] == EMP) &&
			    (board[i + 1][j + 1] == EMP))
				return 2;
		}
	}
	return 0;
}

int legalmove(int i, int j, int lr, const Square &current, const Square (&board)[HT][WD]) {
	if ((i < 0) || (j < 0) || (i > HT - 1) || (j + HT > i + WD)) return 0;
	if (board[i][j] != EMP) return 0;
	if (fall(current, board)) {
		if (i == 0) return 0;
		if ((lr == 0) && (j == 0)) return 0;
		if (board[i - 1][j - 1 + lr] != current) return 0;
		return (board[i][j + (2 * lr) - 1] == EMP);
	} else if (i == HT - 1)
		return (piececount[CCONV(current)] < MAXPIECE);
	else {
		if (board[i + 1][j + lr] != current) return 0;
		return (board[i + 1][j + 1 - lr] != EMP);
	}
}

int countmoves(const Square &plyr, const Square (&board)[HT][WD]) {
	int i, j, lr, sidescheck, fval, moves = 0;
	fval = fall(plyr, board);
	for (i = 0; i < HT; i++) {
		sidescheck = 1 + (fval || (i != HT - 1));
		for (j = 0; j < 1 + i + WD - HT; j++)
			for (lr = 0; lr < sidescheck; lr++)  // left or right
				moves += (legalmove(i, j, lr, plyr, board));
	}
	return moves;
}

int evaluate(const Square &plyr, const Square (&board)[HT][WD]) {
	int i, j, lr, sidescheck, fval, place = 0, moves = 0;
	fval = fall(plyr, board);
	for (i = 0; i < HT; i++) {
		sidescheck = 1 + (fval || (i != HT - 1));
		for (j = 0; j < 1 + i + WD - HT; j++) {
			place += ((board[i][j] == plyr) ? 1 : 0) *
			         (3 * (HT - i) + std::min(2 + j, 2 + WD - HT + i - j));
			for (lr = 0; lr < sidescheck; lr++)  // left or right
				moves += (legalmove(i, j, lr, plyr, board));
		}
	}
	if (moves == 0)
		return LOSE / 2;
	else if (moves == 1)
		return LOSE / 3;
	else
		return place * (moves + 5);
}

// nega-alpha-beta search,
int search(int depth, int alpha, int beta, int top, Square &current, Square (&board)[HT][WD]) {
	int k, i, j, fval, lr, value, localalpha = alpha, hasmove = 0, nearwin = 0;
	int topsave, sidescheck, bestmoves = 0;

	if (depth) {
		fval = fall(current, board);
		// check if current player has enough up top
		value = 0;
		for (j = 0; j < 1 + WD - HT; j++) value += (board[0][j] == current);
		if (value >= WINCOUNT) {
			if (fval == 1)
				nearwin = 1;
			else
				return WIN;
		}

		// search for empty squares, then checks if they can be moved to
		for (i = HT - 1; i >= !!fval; i--) {
			sidescheck = 1 + (fval || (i != HT - 1));
			// don't check top row in fall situation
			for (j = 0; j < (1 + i + WD - HT); j++)
				for (lr = 0; lr < sidescheck; lr++)  // left or right
					if (legalmove(i, j, lr, current, board)) {
						hasmove = 1;
						// finally time to make the move!
						board[i][j] = current;
						if (fval)
							board[i - 1][j - 1 + lr] = EMP;
						else if (i != HT - 1)
							board[i + 1][j + lr] = EMP;
						else
							piececount[CCONV(current)]++;
						current = SW(current);

						// make the recursive call
						value = -search(depth - 1, -beta, -localalpha + top, 0, current, board);

						// undo move
						current = SW(current);
						board[i][j] = EMP;
						if (fval)
							board[i - 1][j - 1 + lr] = current;
						else if (i != HT - 1)
							board[i + 1][j + lr] = current;
						else
							piececount[CCONV(current)]--;

						if (top && VERBOSE == 2) {
							printf("%d,%d ", i, j);
							if (lr)
								printf("right  ");
							else
								printf("left   ");
							if (value >= localalpha)
								printf("%d", value);
							else
								printf("wrs");
							printf("\n");
						}

						if (value >= beta) {
							if (top) {
								savei = i;
								savej = j;
								savelr = lr;
								savevalue = value;
							}
							return value;
						}
						if (value > localalpha) {
							localalpha = value;
							if (top) {
								bestmoves = 1;
								savei = i;
								savej = j;
								savelr = lr;
								savevalue = value;
							}
						} else if ((top) && (value == localalpha) &&
						           (!(rand() % (++bestmoves)))) {
							savei = i;
							savej = j;
							savelr = lr;
							savevalue = value;
						}
					}
		}
		return (hasmove) ? localalpha : LOSE;
	} else
		return evaluate(current, board) - evaluate(SW(current), board);
}

int undomove(Square &current, Square (&board)[HT][WD]) {
	int i, fval;
	for (i = 0; i < 2; i++) {
		if (movenumber == 0) return 0;
		movenumber--;
		current = SW(current);
		savei = moverecord[movenumber][0];
		savej = moverecord[movenumber][1];
		savelr = moverecord[movenumber][2];
		fval = moverecord[movenumber][3];
		board[savei][savej] = EMP;
		if (fval)
			board[savei - 1][savej - 1 + savelr] = current;
		else if (savei != HT - 1)
			board[savei + 1][savej + savelr] = current;
		else
			piececount[CCONV(current)]--;
	}
	return movenumber;
}

int main() {
	int fval, okmove, winner, i;

	srand(time(0));
	srand48(time(0));  // seed the random number guys

	if ((outpoint = fopen(OUTFILE, "w")) == NULL)
		fprintf(stderr, "Cannot open %s\n", OUTFILE);

	Square current;
	Square board[HT][WD];
	initboard(current, board);
	while ((search(1, LOSE, WIN, 0, current, board) != LOSE) &&
	       (search(1, LOSE, WIN, 0, current, board) != WIN)) {
		if (VERBOSE || HUMAN) printboard(current, board);

		if (HUMAN) {
			if (current == DK) {
				do {
					printf("height of destination (%d to undo)? ", HT);
					scanf("%d", &savei);
					if (savei == HT) {
						movenumber = undomove(current, board);
						goto endofwhile;
					}
					printf("across of destination? ");
					scanf("%d", &savej);
					if (!legalmove(savei, savej, 1, current, board))
						savelr = 0;
					else if (!legalmove(savei, savej, 0, current, board))
						savelr = 1;
					else if ((savei != HT - 1) || (fall(current, board))) {
						printf("from (left=1,right=2?) ");
						scanf("%d", &savelr);
						savelr--;
					}
				} while (!legalmove(savei, savej, savelr, current, board));
			} else {
				search(DEPTH, LOSE, WIN, 1, current, board);
				if (VERBOSE) {
					printf("Computer moves to (%d,%d) from the ", savei, savej);
					if (savelr == 0)
						printf("left. Value %d.\n", savevalue);
					else
						printf("right. Value %d.\n", savevalue);
				}
			}
		} else {
			// if you've provably lost, don't do a deep search
			if (savevalue == WIN) {
				search(1, LOSE, WIN, 1, current, board);
				savevalue = LOSE;
			} else
				search(DEPTH, LOSE, WIN, 1, current, board);
			if (VERBOSE) {
				printf("Computer moves to (%d,%d) from the ", savei, savej);
				if (savelr == 0)
					printf("left. Value %d.", savevalue);
				else
					printf("right. Value %d.", savevalue);
			}
		}

		if (current == FIRST) fprintf(outpoint, "%d. ", 1 + movenumber / 2);
		fprintf(outpoint, "%d,%d ", savei, savej);
		if (!fval && savei == HT - 1)
			fprintf(outpoint, "place, ");
		else {
			if (savelr)
				fprintf(outpoint, "right, ");
			else
				fprintf(outpoint, "left, ");
		}

		// if the computer is playing, record how well it thinks it's doing
		if (!HUMAN || (current != DK))
			fprintf(outpoint, "%d\t\t", savevalue);
		else
			fprintf(outpoint, "\t\t");

		if (current != FIRST) fprintf(outpoint, "\n");

		// make the move
		fval = fall(current, board);
		board[savei][savej] = current;
		if (fval)
			board[savei - 1][savej - 1 + savelr] = EMP;
		else if (savei != HT - 1)
			board[savei + 1][savej + savelr] = EMP;
		else
			piececount[CCONV(current)]++;
		current = SW(current);
		moverecord[movenumber][0] = savei;
		moverecord[movenumber][1] = savej;
		moverecord[movenumber][2] = savelr;
		moverecord[movenumber][3] = fval;
		movenumber++;
	endofwhile:
		printf("movenumber=%d\n", movenumber);
	}

	if (VERBOSE || HUMAN) {
		printboard(current, board);
		printf("\n");
	}
	winner = (search(1, LOSE, WIN, 0, current, board) == LOSE) ? SW(current) : current;
	printf("The winner is ");
	printsquare(winner);
	printf("!!\n");
	fprintf(outpoint, "\n");
	fclose(outpoint);
}
