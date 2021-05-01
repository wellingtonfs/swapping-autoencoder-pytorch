#include <stdio.h>

int main(){

    int W, F, P, S;

    while(1){
        printf("W F S P: ");
        scanf("%d %d %d %d", &W, &F, &S, &P);

        int res = (( W - F + (2*P) ) / S) + 1;

        printf("(%d,%d)\n", res, res);
    }

    return 0;
}