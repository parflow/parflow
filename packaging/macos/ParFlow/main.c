/*
 * ParFlow.app launcher — resolves Resources/parflow and execs parflow.
 * CLI users typically call Contents/Resources/parflow/bin/parflow directly.
 */
#include <mach-o/dyld.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int resolve_parflow_dir(char *out, size_t outlen)
{
  char exec_path[PATH_MAX];
  uint32_t size = sizeof(exec_path);
  if (_NSGetExecutablePath(exec_path, &size) != 0)
    return -1;

  char exec_copy[PATH_MAX];
  strncpy(exec_copy, exec_path, sizeof(exec_copy));
  exec_copy[sizeof(exec_copy) - 1] = '\0';

  char *macos_dir = dirname(exec_copy);
  char contents[PATH_MAX];
  snprintf(contents, sizeof(contents), "%s/..", macos_dir);

  char contents_real[PATH_MAX];
  if (!realpath(contents, contents_real))
    return -1;

  snprintf(out, outlen, "%s/Resources/parflow", contents_real);
  return 0;
}

int main(int argc, char *argv[])
{
  char pfdir[PATH_MAX];
  if (resolve_parflow_dir(pfdir, sizeof(pfdir)) != 0) {
    fprintf(stderr, "ParFlow.app: could not locate Resources/parflow\n");
    return 1;
  }

  if (setenv("PARFLOW_DIR", pfdir, 1) != 0) {
    perror("setenv PARFLOW_DIR");
    return 1;
  }

  char parflow_bin[PATH_MAX];
  snprintf(parflow_bin, sizeof(parflow_bin), "%s/bin/parflow", pfdir);

  char **new_argv = malloc((size_t)(argc + 1) * sizeof(char *));
  if (!new_argv) {
    perror("malloc");
    return 1;
  }
  new_argv[0] = parflow_bin;
  for (int i = 1; i < argc; i++)
    new_argv[i] = argv[i];
  new_argv[argc] = NULL;

  execv(parflow_bin, new_argv);
  perror("execv parflow");
  free(new_argv);
  return 1;
}
