#include <fca/fca.h>
#include "parflow_config.h"
#include <messages.h>
#include <stdio.h>
#include <stdlib.h>  // for rand()
#include <assert.h>
#include <unistd.h>
#include <math.h>


#ifdef __DEBUG
#define D(x ...) printf("aaaaaaaa "); printf(x); printf(" %s:%d\n", __FILE__, __LINE__)
#else
#define D(...)
#endif

// TODO: there must be a very similar code in the pfsimulator code. use it?!
void setInBox(double *data, SteerMessageMetadata const * const s, double value, double lower_x, double lower_y, double lower_z,
              double upper_x, double upper_y, double upper_z)
{
  const int dx = 10;
  const int dy = 10;
  const int dz = 1;

  // TODO:  better information transmission..., too much hard coded
  for (int x = (int)lower_x / dx; x < (int)upper_x / dx; ++x)
  {
    for (int y = (int)lower_y / dy; y < (int)upper_y / dy; ++y)
    {
      for (int z = (int)lower_z / dz; z < (int)upper_z / dz; ++z)  // TODO: probably we can use a classic boxloop here ...
      {
        if (x >= s->ix && x < s->ix + s->nx &&
            y >= s->iy && y < s->iy + s->ny &&
            z >= s->iz && z < s->iz + s->nz)
        {
          size_t index = x - s->ix + (y - s->iy) * s->nx + (z - s->iz) * s->nx * s->ny;
          data[index] = value;
        }
      }
    }
  }
}

/**
 * returns the value at x, y, z in merged grid messages
 */
double findValueAt(void *gridMessages, int x, int y, int z)
{
  // TODO: not pointer safe!
  int xn, yn, zn;  // normalized coords (coords in grid part)
  void *pos = gridMessages;

  while (1)
  {
    GridMessageMetadata *m = (GridMessageMetadata*)pos;
    // prepare pos of next GridMessage...
    pos += sizeof(GridMessageMetadata) + sizeof(double) * m->nx * m->ny * m->nz;
    // calculate normalized coords of requested point
    xn = x - m->ix;
    yn = y - m->iy;
    zn = z - m->iz;
    // is the requested point in this chunk?
    if (xn < 0 || yn < 0 || zn < 0)
      continue;                                // too small.
    if (xn >= m->nx || yn >= m->ny || zn >= m->nz)
      continue;                                            // too big.
    // yes it is...
    size_t index = xn + yn * m->nx + zn * m->nx * m->ny;

    // you'll find it behind the GridMessageMetadata:
    return ((double*)(m + 1))[index];
  }
}

double *initial_pressure;
//dump initial pressure... Very experimental: TODO


int main(int argc, char *argv [])
{
  /***********************
   * init FlowVR Module
   */
  fca_module flowvr = fca_new_empty_module();
  fca_port portPressureIn = fca_new_port("pressureIn", fca_IN, 0, NULL);

  fca_append_port(flowvr, portPressureIn);

  fca_port portSteerOut = fca_new_port("steerOut", fca_OUT, 0, NULL);
  fca_append_port(flowvr, portSteerOut);

#ifdef __DEBUG
  fca_port portLog = fca_new_port("log", fca_OUT, 0, NULL);
  fca_append_port(flowvr, portLog);
  const fca_stamp stampK = fca_register_stamp(portLog, "K", fca_FLOAT);
  const fca_stamp stampE = fca_register_stamp(portLog, "E", fca_FLOAT);
  const fca_stamp stampM = fca_register_stamp(portLog, "M", fca_FLOAT);
#endif

  if (!fca_init_module(flowvr))
  {
    printf("ERROR : init_module failed!\n");
  }

  D("now waiting\n");
  while (fca_wait(flowvr))
  {
    fca_message msg = fca_get(portPressureIn);
// do something like this instead:
//ParseMergedMessage(portPressureIn, setSnapshot, (void*)sim);
// see content TODO
// do steering.
    GridMessageMetadata *m;
    m = (GridMessageMetadata*)fca_get_read_access(msg, 0);

    D("analyzing timestep %f", m->time);
    // only have a look after the rain after....
    int timestep = (int)round(m->time / 15.);
    if (timestep == 0)
    {
      // dump pressure...
      // TODO: put into function

      initial_pressure = (double*)malloc(sizeof(double) * m->grid.nX * m->grid.nY * m->grid.nZ);
      void *buffer = (void*)m;
      void *end = buffer + fca_get_segment_size(msg, 0);
      while (buffer < end)
      {
        buffer += sizeof(GridMessageMetadata);
        double* data = (double*)buffer;

        // TODO:  use reader module here!
        for (int z = 0; z < m->nz; ++z)
        {
          for (int y = 0; y < m->ny; ++y)
          {
            int index = m->ix + (y + m->iy) * m->grid.nX + (z + m->iz) * m->grid.nX * m->grid.nY;
            memcpy(initial_pressure + index, data, m->nx * sizeof(double));
            data += m->nx;
          }
        }

        buffer += sizeof(double) * m->nx * m->ny * m->nz;
        m = (GridMessageMetadata*)buffer;
      }
    }
    else if ((timestep - 5) % 10 != 0)
    {
      fca_free(msg);
      D("NO");
      continue;
    }
    // timestep == 5, 15, 25, ...

    D("YES");
    SteerMessageMetadata s;
    s.ix = 0;
    s.iy = 0;
    s.iz = 0;
    s.nx = m->grid.nX;
    s.ny = m->grid.nY;
    s.nz = m->grid.nZ;


    double measured_value = findValueAt((void*)m, 0, 0, m->grid.nZ - 1);

    fca_free(msg);

    size_t size = sizeof(SteerMessageMetadata) + sizeof(double) * s.nx * s.ny * s.nz;
    void *parameter = malloc(size);
    memcpy(parameter, (void*)&s, sizeof(SteerMessageMetadata));
    double *operand = (double*)(parameter + sizeof(SteerMessageMetadata));

    double H1_Lower_X = 0;
    double H1_Lower_Y = 0;
    double H1_Lower_Z = 20;
    double H1_Upper_X = 10;
    double H1_Upper_Y = 500;
    double H1_Upper_Z = 24;
    double H2_Lower_X = 0;
    double H2_Lower_Y = 0;
    double H2_Lower_Z = 15;
    double H2_Upper_X = 10;
    double H2_Upper_Y = 500;
    double H2_Upper_Z = 20;
    double H3_Lower_X = 0;
    double H3_Lower_Y = 0;
    double H3_Lower_Z = 10;
    double H3_Upper_X = 10;
    double H3_Upper_Y = 500;
    double H3_Upper_Z = 15;
    double H4_Lower_X = 0;
    double H4_Lower_Y = 0;
    double H4_Lower_Z = 5;
    double H4_Upper_X = 10;
    double H4_Upper_Y = 500;
    double H4_Upper_Z = 10;
    double H5_Lower_X = 0;
    double H5_Lower_Y = 0;
    double H5_Lower_Z = 0;
    double H5_Upper_X = 10;
    double H5_Upper_Y = 500;
    double H5_Upper_Z = 5;

    // set standard perm
    // TODO: later this must be all parametrized!
    static double K;
    static double last_error = 0.;
    static double integral = 0.;

    const double setpoint = 0.;  // want it to coule...
    const double Kp = 1.0;
    const double Ki = 2.0;
    const double Kd = 2.0;
    const double dt = 1.0;  // TODO: this needs to be changed aswell... (into simulationruns/ times)
    // PID
    {
      double error = setpoint - measured_value;
      integral += error * dt;
      double u = Kp * error + Ki * integral + Kd * (error - last_error) / dt;

      last_error = error;
      K = u; // TODO: += -> = ! // that's the questionTODO
    }


    double K1 = 0.0000001667;
    double K2 = 0.0042;
    double K3 = 0.003;
    double K4 = 0.0042;
    double K5 = 0.0042;

    /*if (K < 0.) K = -K;*/
    if (K < 0.)
      K = 0.;
    /*if (K > 1.) K = 1.;*/

    /*K = 0.;*/

    // change the highest layer...
    /*double K1 = K;*/

#ifdef __DEBUG
    fca_message msgLog = fca_new_message(flowvr, 0);
    float stampData = (float)K;
    float stampError = (float)last_error;
    float stampMeasured = (float)measured_value;
    fca_write_stamp(msgLog, stampK, (void*)&stampData);
    fca_write_stamp(msgLog, stampE, (void*)&stampError);
    fca_write_stamp(msgLog, stampM, (void*)&stampMeasured);
    fca_put(portLog, msgLog);
    fca_free(msgLog);
#endif


    // TODO: we have to set all boxes as steemessagemetadata says we reset all the area...
    setInBox(operand, &s, K1, H1_Lower_X, H1_Lower_Y, H1_Lower_Z,
             H1_Upper_X, H1_Upper_Y, H1_Upper_Z);
    setInBox(operand, &s, K2, H2_Lower_X, H2_Lower_Y, H2_Lower_Z,
             H2_Upper_X, H2_Upper_Y, H2_Upper_Z);
    setInBox(operand, &s, K3, H3_Lower_X, H3_Lower_Y, H3_Lower_Z,
             H3_Upper_X, H3_Upper_Y, H3_Upper_Z);
    setInBox(operand, &s, K4, H4_Lower_X, H4_Lower_Y, H4_Lower_Z,
             H4_Upper_X, H4_Upper_Y, H4_Upper_Z);
    // TODO: do not change in all the size ;)
    setInBox(operand, &s, K5, H5_Lower_X, H5_Lower_Y, H5_Lower_Z,
             H5_Upper_X, H5_Upper_Y, H5_Upper_Z);

    D("Send steer!");
    // Should not change anything for now.
    // TODO: does this work as we are sending multiple actions here...
    SendActionMessage(flowvr, portSteerOut, ACTION_SET,
                      VARIABLE_PERMEABILITY_X, parameter, size);
    SendActionMessage(flowvr, portSteerOut, ACTION_SET,
                      VARIABLE_PERMEABILITY_Y, parameter, size);
    SendActionMessage(flowvr, portSteerOut, ACTION_SET,
                      VARIABLE_PERMEABILITY_Z, parameter, size);

    free(parameter);
    // TODO: SendActionsMessage for multiple actions
    // sth like add action and then send action...

    // TODO: rename into action port (triggerSnap and steeout?) or at least show type in port name??
  }

  fca_free(flowvr);
}
