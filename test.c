#include<stdio.h>
int main()
{
	int n;
	scanf("%d",&n);
	printf("%d\n", n*n);
	char s;
	printf("Wanna try more?[y/n] : ");
	scanf("%c",&s);
	if(s=='y')
	{
		main();
	}
	else if(s=='n')
	{
		return 0;
	}
	else
	{
		printf("wrong input,closing!!!");
		return 0;
	}
}
