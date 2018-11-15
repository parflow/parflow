#include "parflow_config.h"  // for __DEBUG
#include "flowvr/daemon.h"
#include "flowvr/plugins/filter.h"
#include "flowvr/plugd/dispatcher.h"
#include "flowvr/plugd/messagequeue.h"
#include "flowvr/mem/sharedmemorymanager.h"
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <cassert>
#include <messages.h>

// To not load the pfanalyzer library:
GenerateMessageReaderC(Grid);

namespace flowvr
{

  namespace plugins
  {

    using namespace flowvr::plugd;

    /// \brief A filter converting concatenated gridmessages into one big grid message
    /// containing the full grid (nX,nY,nZ)
    /// of timestep t as soon as he got all the necessary data.
    ///
    /// <b>WARNING:</b> we do not check if the grid messages do not overlap when counting
    /// the size!
    /// <b>WARNING:</b> we abort if not all the data necessary were found!
    ///
    /// <b>Input ports:</b>
    /// -  <b>in</b>: Grid messages to be joined.
    ///
    /// <b>Output Ports:</b>
    /// - <b>out</b>: Big grid messages.

    class MergeGridMessages : public Filter
    {
      public:

        MergeGridMessages(const std::string objID);

        virtual ~MergeGridMessages();

        virtual Class* getClass() const;

        virtual flowvr::plugd::Result init(flowvr::xml::DOMElement* xmlRoot, flowvr::plugd::Dispatcher* dispatcher);

        virtual void newMessageNotification(int mqid, int msgnum, const Message& msg, Dispatcher* dispatcher);
        virtual void newStampListSpecification(int mqid, const Message& msg, Dispatcher* dispatcher);

      protected:
        BufferPool poolout;

        virtual void sendPendingOrders(plugd::Dispatcher* dispatcher);
        virtual bool hasStampSpecification();
    };

    using namespace flowvr::xml;

    /// Constructor.
    MergeGridMessages::MergeGridMessages(const std::string objID)
      : Filter(objID)
    {
    }

    MergeGridMessages::~MergeGridMessages()
    {
    }

    flowvr::plugd::Result MergeGridMessages::init(flowvr::xml::DOMElement* xmlRoot, flowvr::plugd::Dispatcher* dispatcher)
    {
      flowvr::plugd::Result result = Filter::init(xmlRoot, dispatcher);
      if (result.error()) return result;

      initInputs(1);
      inputs[0]->setName("in");

      initOutputs(1);
      outputs[0]->setName("out");

      return result;
    }

    void MergeGridMessages::newMessageNotification(int mqid, int msgnum, const Message& msg, Dispatcher* dispatcher)
    {
      sendPendingOrders(dispatcher);
    }

    void MergeGridMessages::newStampListSpecification(int mqid, const Message& msg, Dispatcher* dispatcher)
    {
      // forward specification to out port
#ifdef __DEBUG
      std::cout << objectID()<<": forwarding stamps specification"<<std::endl;
#endif

      //give the Stamplist to the outputmessage queue
      outputs[0]->stamps = inputs[0]->getStampList();
      outputs[0]->newStampSpecification(dispatcher);

      sendPendingOrders(dispatcher);
    }

    bool MergeGridMessages::hasStampSpecification()
    {
      if (!inputs[0]->stampsReceived())
      {
        std::cout << "waiting for stamp specs" << std::endl;

        return false;
      }
      return true;
    }

    void MergeGridMessages::sendPendingOrders(plugd::Dispatcher* dispatcher)
    { // MAIN FILTER FUNCTION

      if (!hasStampSpecification()) return; // still waiting for stamps specification
      // Read in messages...

      for(;;)
      {
        if (!inputs[0]->frontMsg().valid())
        {
          return;
        }

        const Message &msg = inputs[0]->frontMsg();

        // Skip stamp messages
        if (msg.getType() == Message::STAMPS)
        {
          return;
        }

        double last_time       = -1.;            /// current time
        size_t last_size       = -1;             /// size to wait for
        size_t size            = 0;              /// size where we are
        Variable last_variable = VARIABLE_LAST;  /// the variable we wait for.

        size_t s = msg.data.getSize();

        // construct big grid message and send it!
        BufferWrite newdata;
        double *data = NULL;

        const void* start = msg.data.readAccess();
        const void* buffer = start;
        const void* end = buffer + s;
        while (buffer < end)
        {
          GridMessage gm = ReadGridMessage(buffer);
          if (gm.m->variable != last_variable || gm.m->time != last_time ||
              gm.m->grid.nX * gm.m->grid.nY * gm.m->grid.nZ != last_size)
          {
            // only the first message may do this!
            assert(buffer == start);

            // check metadata
            assert(gm.m->variable >= 0);
            assert(gm.m->variable < VARIABLE_LAST);
            assert(gm.m->grid.nX > 0);
            assert(gm.m->grid.nY > 0);
            assert(gm.m->grid.nZ > 0);
            assert(gm.m->ix >= 0);
            assert(gm.m->iy >= 0);
            assert(gm.m->iz >= 0);
            assert(gm.m->nx > 0);
            assert(gm.m->ny > 0);
            assert(gm.m->nz > 0);

            last_variable = gm.m->variable;
            last_time = gm.m->time;
            last_size = gm.m->grid.nX * gm.m->grid.nY * gm.m->grid.nZ;

            // Allocate data for new message
            newdata = poolout.alloc(getAllocator(), last_size * sizeof(double) +
                sizeof(GridMessageMetadata));

            // Populate metadata
            GridMessageMetadata *metadata = (GridMessageMetadata*) newdata.writeAccess();
            memcpy((void*) metadata, gm.m, sizeof(GridMessageMetadata));
            metadata->nx = metadata->grid.nX;
            metadata->ny = metadata->grid.nY;
            metadata->nz = metadata->grid.nZ;
            metadata->ix = 0;
            metadata->iy = 0;
            metadata->iz = 0;

            // Prepare pointer for data
            data = (double*) (metadata + 1);
          }

          for (int z = 0; z < gm.m->nz; ++z)
          {
            for (int y = 0; y < gm.m->ny; ++y)
            {
              int index = gm.m->ix + (y + gm.m->iy) * gm.m->grid.nX + (z + gm.m->iz) *
                gm.m->grid.nX * gm.m->grid.nY;
              memcpy(data + index, gm.data, gm.m->nx * sizeof(double));
              gm.data += gm.m->nx;
            }
          }

          // Accumulate how much we read out already
          size += gm.m->nx * gm.m->ny * gm.m->nz;

#ifdef __DEBUG
          std::cout<<objectID()<<": found a grid message of "<< msg.data.getSize() <<
            "bytes" << std::endl;
#endif
          if (size < last_size)
          {
#ifdef __DEBUG
            std::cout<<objectID()<<": reading for more grid segments for t="<<last_time<<std::endl;
#endif
          }
          else
          {
            assert(size == last_size);


            MessagePut newmsg;
            newmsg.data = newdata;

            newmsg.stamps.clone(msg.stamps, &inputs[0]->getStampList());
            outputs[0]->put(newmsg, dispatcher);

            newmsg.clear();
            newdata.clear();
          }
          buffer += sizeof(GridMessageMetadata) + gm.m->nx * gm.m->ny * gm.m->nz *
            sizeof(double);
        }
        if (buffer > end)
        {
          std::cout << objectID() << ": Probably a message parsing error occured!" <<
            std::endl;
        }
        inputs[0]->eraseFront();
      }
    }

    flowvr::plugd::GenClass<MergeGridMessages> MergeGridMessagesClass("flowvr.plugins.MergeGridMessages", // name
        "", // description
        &flowvr::plugins::FilterClass
        );

    Class* MergeGridMessages::getClass() const
    {
      return &MergeGridMessagesClass;
    }

  } // namespace plugins

} // namespace flowvr
