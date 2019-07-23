/* Copyright 2008 Ohio Supercomputer Center */

/* Distribution of this program is governed by the GNU GPL.  See
   ../COPYING for details. */

/*
   Usage:  mpiexec [-n #] [arg] parallel-command-processor cfgfile
      OR:  mpiexec [-n #] [arg] parallel-command-processor << EOF
           cmd1
           cmd2
           ...
           cmdM
           EOF
 */

/* Modification by Cicada Dennis of Indiana University, Research Technologies
 * 2019-07-22, We found that the code crashed during the termination/cleanup
 * phase of the program. It would not affect people's results, since it only
 * happened after their code was finished running.
 * It seems to have been caused by calculating MPI_Request addresses in a 
 * way that caused MPI_Isend() commands to write outside of the the bounds 
 * of the allocated MPI_Request space.
*/

#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

//#define DEBUG	1
#define REQUEST_TAG  0
#define STATUS_TAG   1
#define DATA_TAG     2

void remove_EOL(char *string)
{
  int len = strlen(string);
  for ( int i=0 ; i<len ; i++ )
    {
      if ( string[i]=='\n' ) string[i]=='\0';
    }
}

void minion(int rank)
{
  char cmd[LINE_MAX+1];
  int cont;
  MPI_Status mystat;
  int ncmds = 0;

  /* setup */
  cont = 1;

  /* initial distribution */
  MPI_Barrier(MPI_COMM_WORLD);

  /* main loop */
  while ( cont==1 )
    {
      int nbytes;
      int retcode;
      memset(cmd,'\0',(size_t)LINE_MAX);
      MPI_Recv(cmd,LINE_MAX,MPI_CHAR,0,DATA_TAG,MPI_COMM_WORLD,&mystat);
      /* do your thing */
      if ( strlen(cmd)>0 )
	{
#ifdef DEBUG
	  printf("Rank %d:  executing \"%s\"\n",rank,cmd);
#endif /* DEBUG */
	  retcode = system(cmd);
#ifdef DEBUG
	  ncmds++;
#endif /* DEBUG */
	}
      MPI_Send(&retcode,1,MPI_INT,0,REQUEST_TAG,MPI_COMM_WORLD);
      MPI_Recv(&cont,1,MPI_INT,0,STATUS_TAG,MPI_COMM_WORLD,&mystat);
    }

  /* cleanup */
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG
  MPI_Reduce(&ncmds,NULL,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
#endif /* DEBUG */
}

void mastermind(int nminions, FILE *input)
{
  int cont = 1;
  int stop = 0;
  int ncmds = 0;
  char cmd[LINE_MAX+1];

  /* setup */

  /* initial distribution */
  MPI_Barrier(MPI_COMM_WORLD);
  while ( ncmds<nminions && !feof(input) )
    {
      memset(cmd,'\0',(size_t)LINE_MAX);
      fgets(cmd,LINE_MAX,input);
      while ( ( strlen(cmd)==0 || cmd[0]=='#' ) && !feof(input) )
	{
	  memset(cmd,'\0',(size_t)LINE_MAX);
	  fgets(cmd,LINE_MAX,input);
	  if ( strlen(cmd)>0 ) remove_EOL(cmd);
	}
      if ( strlen(cmd)>0 )
	{
	  MPI_Send(cmd,strlen(cmd)-1,MPI_CHAR,ncmds+1,DATA_TAG,MPI_COMM_WORLD);
	  ncmds++;
	}
    }

  /* main loop */
  while ( !feof(input) )
    {
      int retcode;
      int next;
      MPI_Status mystat;

      memset(cmd,'\0',(size_t)LINE_MAX);
      fgets(cmd,LINE_MAX,input);
      while ( ( strlen(cmd)==0 || cmd[0]=='#') && !feof(input) )
	{
	  memset(cmd,'\0',(size_t)LINE_MAX);
	  fgets(cmd,LINE_MAX,input);
	  if ( strlen(cmd)>0 ) remove_EOL(cmd);
	}
      if ( strlen(cmd)>0 )
	{
          /* There is another command to run. 
           * Find a rank that has completed its task. */
          MPI_Recv(&retcode,1,MPI_INT,MPI_ANY_SOURCE,REQUEST_TAG,MPI_COMM_WORLD,
	       &mystat);
          next = mystat.MPI_SOURCE;
 #ifdef DEBUG
          printf("Rank 0:  rank %d returned code %d\n",next,retcode);
#endif /* DEBUG */
          /* Tell the rank to continue and send it the command to run. */
          MPI_Send(&cont,1,MPI_INT,next,STATUS_TAG,MPI_COMM_WORLD);
	  MPI_Send(cmd,strlen(cmd)-1,MPI_CHAR,next,DATA_TAG,
		   MPI_COMM_WORLD);
	}
    }

  /* cleanup */
  MPI_Request *req;
  req = (MPI_Request *)calloc((size_t)(2*nminions),sizeof(MPI_Request));
  // 2019-07-22, Change by Cicada Dennis
  // On IU systems something about the way the address for req was calculated
  // in MPI_Isend() commands was causing seg faults or writing out of bounds.
  // Changed the index variable i to start at 0 instead of 1, but probs were
  // still happening.
  // Changed the MPI_Isend() commands to use subscripting for the
  // req address value and it fixed problem.
  for ( int i = 0 ; i < nminions ; i++ )
    {
      char exitcmd[] = "exit";
      int rank = i+1;

      //MPI_Isend(exitcmd,strlen(exitcmd),MPI_CHAR,i,DATA_TAG,MPI_COMM_WORLD,
      //	req+(size_t)(2*i)*sizeof(MPI_Request));
      MPI_Isend(exitcmd,strlen(exitcmd),MPI_CHAR,rank,DATA_TAG,MPI_COMM_WORLD,
              &(req[2*i]));
      //MPI_Isend(&stop,1,MPI_INT,i,STATUS_TAG,MPI_COMM_WORLD,
      //	req+(size_t)(2*i+1)*sizeof(MPI_Request));
      MPI_Isend(&stop,1,MPI_INT,rank,STATUS_TAG,MPI_COMM_WORLD,
              &(req[2*i+1]));
    }
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG
  ncmds=0;
  MPI_Reduce(&stop,&ncmds,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
  printf("Executed %ld commands\n",ncmds);
#endif /* DEBUG */
  free(req);
}

int main(int argc, char *argv[])
{
  int rank;
  int nproc;
  FILE *input;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  if ( nproc<2 && rank==0 )
    {
      fprintf(stderr,"%s:  At least 2 MPI processes required!\n",argv[0]);
      exit(-1);
    }
  if ( rank==0 )
    {
      if ( argc>1 )
	{
	  input = fopen(argv[1],"r");
	}
      else
	{
	  input = stdin;
	}
      mastermind(nproc-1,input);
    }
  else
    {
      minion(rank);
    }
  MPI_Finalize();
  return(0);
}
