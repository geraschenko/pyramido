#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DEPTH 8
#define HUMAN 1  // are you gunna be there
#define OUTFILE "out.txt"  // where do I record the game
#define WEIGHTFILE "weights.pyr"  // from whence to load/where to store weights
#define VERBOSE 1  //   0 = only output winner
// 1 = "normal" output
// 2 = info about all evaluated moves
#define FIRST LT  // who plays first
#define ALPHA 1  // learning rate

#define ILL ((square)2)
#define EMP ((square)0)
#define DK ((square)1)  // Player ``Dark'' is X
#define LT ((square)-1)  // Player ``Light'' is O
#define SW(x) (-x)
#define CCONV(x) ((1 - x) / 2)  // 0 for DK and 1 for LT

#define HT 5
#define WD 8
#define MAXPIECE 13
#define WINCOUNT 2

#define WIN 10000
#define LOSE (-WIN)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CAP(x) (((x) > 0) ? MIN(x, 1) : MAX(x, -1))
#define STRETCH (10.0)
#define SIGMA(x) (-1.0 + 2.0 / (1.0 + exp(-(x) / STRETCH)))
#define DSIGMAAT(x)                              \
	(0.5 * (1.0 - (x / STRETCH) * (x / STRETCH)) / \
	 STRETCH)  // derivative of SIGMA at SIGMA^{-1}(x)

#define C2(a) (((a) * (a) - (a)) / 2)
#define SQCNT (C2(WD + 1) - C2(WD - HT + 1))  // number of board squares

#define HNUM 3  // number of helper inputs
// fval, numberofmoves(current), numberofmoves(SW(current))

#define INLAYER (SQCNT + HNUM)  // input layer
#define HLAYER1 (1)  // hidden layers
#define HLAYER2 (1)
#define NODES (HLAYER1 + HLAYER2 + 1)
#define PLACE1(x) ((x)-INLAYER)
#define PLACE2(x) ((x)-INLAYER - HLAYER1)

double weight[NODES][MAX(MAX(INLAYER, HLAYER1), HLAYER2)];
// weight[i][j] is the weight of the j-th input of the i-th node
double savequality, prospects;

typedef char square;

square board[HT][WD];
square current;
int piececount[2];
int savei, savej, savelr, savevalue;
FILE *outpoint, *weightpoint;
int movenumber = 0;
int moverecord[100][4];

void printsquare(square x) {
	if (x == EMP) {
		printf(" ");
		return;
	}
	if (x == DK) {
		printf("X");
		return;
	}
	if (x == LT) {
		printf("O");
		return;
	} else
		printf("?");
}

void printboard() {
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

void initboard() {
	int i, j;
	current = FIRST;
	piececount[0] = 0;
	piececount[1] = 0;
	for (i = 0; i < HT; i++)
		for (j = 0; j < WD; j++) board[i][j] = (j + HT > i + WD) ? ILL : EMP;
}

void initweights() {
	int i, j;
	if ((weightpoint = fopen(WEIGHTFILE, "rb")) == NULL) {
		printf("Cannot get weights from %s. Loading random weights.\n", WEIGHTFILE);
		for (i = 0; i < NODES; i++) {
			for (j = 0; j < MAX(MAX(INLAYER, HLAYER1), HLAYER2); j++)
				weight[i][j] = 10 * drand48() - 5;
		}
	} else {
		if (fread(weight,
		          (NODES * MAX(MAX(INLAYER, HLAYER1), HLAYER2)) * sizeof(double), 1,
		          weightpoint) != 1) {
			fprintf(stderr, "Error reading %s.\n", WEIGHTFILE);
			exit(1);
		} else {
			if (VERBOSE || HUMAN) printf("Loading weights from %s.\n", WEIGHTFILE);
			fclose(weightpoint);
		}
	}
}

void saveweights() {
	if ((weightpoint = fopen(WEIGHTFILE, "wb")) == NULL)
		printf("Cannot open %s.\n", WEIGHTFILE);
	if (fwrite(weight,
	           (NODES * MAX(MAX(INLAYER, HLAYER1), HLAYER2)) * sizeof(double), 1,
	           weightpoint) != 1)
		printf("Error writing to %s.\n", WEIGHTFILE);
	fclose(weightpoint);
}

void printweights() {
	int i, j;
	for (i = 0; i < NODES; i++) {
		printf("\nweights for node %d:", i);
		for (j = 0; j < MAX(MAX(INLAYER, HLAYER1), HLAYER2); j++)
			printf("%lf, ", weight[i][j]);
		printf("\n");
	}
}

// must current player fall this move?
// returns 1 if have to fall from top, 2 if have to fall elsewhere, 0 otherwise
int fall() {
	int i, j;

	// check top row first
	for (j = 0; j < WD - HT + 1; j++)
		if ((board[0][j] == current) && (board[1][j] == EMP) &&
		    (board[1][j + 1] == EMP))
			return 1;
	for (i = HT - 2; i > 0; i--)  // skip bottom row
		for (j = 0; j < WD - HT + 1 + i; j++)
			if ((board[i][j] == current) && (board[i + 1][j] == EMP) &&
			    (board[i + 1][j + 1] == EMP))
				return 2;
	return 0;
}

int countmoves(square plyr) {
	int i, j, lr, sidescheck, fval, moves = 0;
	square who = current;

	current = plyr;  // temporarily set the current player to plyr

	fval = fall();
	for (i = 0; i < HT; i++) {
		sidescheck = 1 + (fval || (i != HT - 1));
		for (j = 0; j < 1 + i + WD - HT; j++)
			for (lr = 0; lr < sidescheck; lr++)  // left or right
				moves += (legalmove(i, j, lr, fval));
	}
	current = who;  // set things straight again
	return moves;
}

int evaluate(square plyr) {
	int i, j, lr, sidescheck, fval, place = 0, moves = 0;
	square who = current;

	current = plyr;  // temporarily set the current player to plyr

	fval = fall();
	for (i = 0; i < HT; i++) {
		sidescheck = 1 + (fval || (i != HT - 1));
		for (j = 0; j < 1 + i + WD - HT; j++) {
			place += ((board[i][j] == current) ? 1 : 0) *
			         (3 * (HT - i) + MIN(2 + j, 2 + WD - HT + i - j));
			for (lr = 0; lr < sidescheck; lr++)  // left or right
				moves += (legalmove(i, j, lr, fval));
		}
	}
	current = who;  // set things straight again
	if (moves == 0)
		return LOSE / 2;
	else if (moves == 1)
		return LOSE / 3;
	else
		return place * (moves + 5);
}

int legalmove(int i, int j, int lr, int fval) {
	if ((i < 0) || (j < 0) || (i > HT - 1) || (j + HT > i + WD)) return 0;
	if (board[i][j] != EMP) return 0;
	if (fval) {
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

// nega-alpha-beta search,
int search(depth, alpha, beta, top) {
	int k, i, j, fval, lr, value, localalpha = alpha, hasmove = 0, nearwin = 0;
	int topsave, sidescheck, bestmoves = 0;

	if (depth) {
		fval = fall();
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
					if (legalmove(i, j, lr, fval)) {
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
						value = -search(depth - 1, -beta, -localalpha + top, 0);

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
		return evaluate(current) - evaluate(SW(current));
}

double quality(char update, double desired) {
	int i, j, k;
	double sum;
	double output[INLAYER + NODES];
	double error[INLAYER + NODES];
	double presigma[NODES];

	output[0] = (double)fall();
	output[1] = (double)(countmoves(current));
	output[2] = (double)(countmoves(SW(current)));
	k = HNUM;
	for (i = 0; i < HT; i++)
		for (j = 0; j < 1 + i + WD - HT; j++) {
			output[k] = (double)(current * board[i][j]);
			k++;
		}
	// compute outputs for first layer
	for (i = 0; i < HLAYER1; i++) {
		sum = 0.0;
		for (j = 0; j < INLAYER; j++) sum += weight[i][j] * output[j];
		output[INLAYER + i] = SIGMA(sum);
		presigma[i] = sum;
	}
	// compute outputs for second layer
	for (i = HLAYER1; i < HLAYER1 + HLAYER2; i++) {
		sum = 0;
		for (j = INLAYER; j < INLAYER + HLAYER1; j++)
			sum += weight[i][PLACE1(j)] * output[j];
		output[INLAYER + i] = SIGMA(sum);
		presigma[i] = sum;
	}
	// compute outputs for output node
	sum = 0;
	for (j = INLAYER + HLAYER1; j < INLAYER + HLAYER1 + HLAYER2; j++)
		sum += weight[NODES - 1][PLACE2(j)] * output[j];
	output[INLAYER + NODES - 1] = SIGMA(sum);
	presigma[NODES - 1] = sum;

	if (update) {
		error[INLAYER + NODES - 1] = output[INLAYER + NODES - 1] - desired;
		// compute errors for the second layer and update the weights to final node
		for (j = INLAYER + HLAYER1; j < INLAYER + HLAYER1 + HLAYER2; j++) {
			error[j] = error[INLAYER + NODES - 1] *
			           DSIGMAAT(output[INLAYER + NODES - 1]) *
			           weight[NODES - 1][PLACE2(j)];
			weight[NODES - 1][PLACE2(j)] -= CAP(ALPHA * error[j]);
		}
		// compute errors for the first layer and update the weights to second layer
		for (i = INLAYER + HLAYER1; i < INLAYER + HLAYER1 + HLAYER2; i++)
			for (j = INLAYER; j < INLAYER + HLAYER1; j++) {
				error[j] = 0.0;
				for (k = HLAYER1; k < HLAYER1 + HLAYER2; k++)
					error[j] += error[INLAYER + k] * DSIGMAAT(output[INLAYER + k]) *
					            weight[k][PLACE1(j)];
				for (k = HLAYER1; k < HLAYER1 + HLAYER2; k++)
					weight[k][PLACE1(j)] -= CAP(ALPHA * error[j]);
			}
		// compute errors for the input layer and update the weights to first layer
		for (i = INLAYER; i < INLAYER + HLAYER1; i++)
			for (j = 0; j < INLAYER; j++) {
				error[j] = 0.0;
				for (k = 0; k < HLAYER1; k++)
					error[j] +=
					    error[INLAYER + k] * DSIGMAAT(output[INLAYER + k]) * weight[k][j];
				for (k = 0; k < HLAYER1; k++) weight[k][j] -= CAP(ALPHA * error[j]);
			}
	}
	for (i = 0; i < NODES; i++) printf("E%d=%lf, ", i, error[i + INLAYER]);
	printf("\n");
	for (i = 0; i < NODES; i++) printf("W%d=%lf, ", i, weight[i][0]);
	printf("\n");
	return output[INLAYER + NODES - 1];
}

int undomove() {
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

	initboard();
	// maybe randomize the board
	//   int j;
	//   for(i=0; i<HT; i++)
	//     for(j=0; j<WD-HT+1+i; j++)
	//       board[i][j] = rand()%3 -1;

	initweights();

	//   printf("\n");
	//   for(i=0; i<500; i++)
	//     quality(1,0.3);
	//   // printweights();
	//   printf("%lf \n", quality(1,0.3));
	//
	//   saveweights();
	//
	//   return 0;

	while ((search(1, LOSE, WIN, 0) != LOSE) &&
	       (search(1, LOSE, WIN, 0) != WIN)) {
		if (VERBOSE || HUMAN) printboard();

		if (HUMAN) {
			if (current == DK) {
				do {
					printf("height of destination (%d to undo)? ", HT);
					scanf("%d", &savei);
					if (savei == HT) {
						movenumber = undomove();
						goto endofwhile;
					}
					printf("across of destination? ");
					scanf("%d", &savej);
					if (!legalmove(savei, savej, 1, fall()))
						savelr = 0;
					else if (!legalmove(savei, savej, 0, fall()))
						savelr = 1;
					else if ((savei != HT - 1) || (fall())) {
						printf("from (left=1,right=2?) ");
						scanf("%d", &savelr);
						savelr--;
					}
				} while (!legalmove(savei, savej, savelr, fall()));
			} else {
				search(DEPTH, LOSE, WIN, 1);
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
				search(1, LOSE, WIN, 1);
				savevalue = LOSE;
			} else
				search(DEPTH, LOSE, WIN, 1);
			if (VERBOSE) {
				printf("Computer moves to (%d,%d) from the ", savei, savej);
				if (savelr == 0)
					printf("left. Value %d.", savevalue);
				else
					printf("right. Value %d.", savevalue);
				printf(" Quality %lf.\n", WIN * quality(0, 0));
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

		// some crufty learning
		if (!HUMAN) {
			savequality = quality(0, 0);
			prospects = savevalue / ((double)WIN);
			quality(1, savequality + ALPHA * prospects);
		}

		// make the move
		fval = fall();
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
		printboard();
		printf("\n");
	}
	winner = (search(1, LOSE, WIN, 0) == LOSE) ? SW(current) : current;
	printf("The winner is ");
	printsquare(winner);
	printf("!!\n");
	fprintf(outpoint, "\n");
	fclose(outpoint);
	saveweights();
}
