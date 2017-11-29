/******* COPYRIGHT ************************************************
 *                                                                 *
 *                             FlowVR                              *
 *                     Daemon and Base Plugins                     *
 *                                                                 *
 *-----------------------------------------------------------------*
 * COPYRIGHT (C) 2003-2011                by                       *
 * INRIA and                                                       *
 * Laboratoire d'Informatique Fondamentale d'Orleans               *
 * (FRE 2490) ALL RIGHTS RESERVED.                                 *
 *                                                                 *
 * This source is covered by the GNU GPL, please refer to the      *
 * COPYING file for further information.                           *
 *                                                                 *
 *-----------------------------------------------------------------*
 *                                                                 *
 *  Original Contributors:                                         *
 *    Jeremie Allard,                                              *
 *    Ronan Gaugne,                                                *
 *    Valerie Gouranton,                                           *
 *    Loick Lecointre,                                             *
 *    Sebastien Limet,                                             *
 *    Bruno Raffin,                                                *
 *    Sophie Robert,                                               *
 *    Emmanuel Melin.                                              *
 *                                                                 *
 *******************************************************************
 *                                                                 *
 * File: src/plugins/flowvr.plugins.MergeItExt.cpp                    *
 *                                                                 *
 * Contacts:                                                       *
 *  12/10/2003 Jeremie Allard <Jeremie.Allard@imag.fr>             *
 *                                                                 *
 ******************************************************************/
#include "parflow_config.h"  // for __DEBUG
#include "flowvr/daemon.h"
#include "flowvr/plugins/filter.h"
#include "flowvr/plugd/dispatcher.h"
#include "flowvr/plugd/messagequeue.h"
#include "flowvr/mem/sharedmemorymanager.h"
#include <iostream>
#include <sstream>
#include <unistd.h>

namespace flowvr
{

  namespace plugins
  {

    using namespace flowvr::plugd;

    /// \brief A filter which sends ALWAYS a message (sometimes containing just a byte
    /// with random content) if order is called. If possible the Message consists of the
    /// concatenation of all messages waiting on the in-ports. The stamps of the first
    /// found message are hereby copied.
    ///
    /// <b>Init parameters:</b>
    /// \<nbIn\>1\<nbIn\>  <i>optional</i>:  number of input ports.  default 1
    //
    /// <b>Input ports:</b>
    /// -  <b>in</b>: Messages to be filtered.
    /// -  <b>order</b>:  Filtering orders  (from a  synchronizer  such as
    /// flowvr::plugins::GreedySynchronizor).
    ///
    /// <b>Output Ports:</b>
    /// - <b>out</b>: Filtered messages.

    class MergeItExt : public Filter
    {
      public:

        MergeItExt(const std::string objID);

        virtual ~MergeItExt();

        virtual Class* getClass() const;

        virtual flowvr::plugd::Result init(flowvr::xml::DOMElement* xmlRoot, flowvr::plugd::Dispatcher* dispatcher);

        virtual void newMessageNotification(int mqid, int msgnum, const Message& msg, Dispatcher* dispatcher);
        virtual void newStampListSpecification(int mqid, const Message& msg, Dispatcher* dispatcher);

        enum {
          IDPORT_ORDER=0
        };

      protected:
        int nbIn;  /// number of input ports

        BufferPool poolout;

        virtual void sendPendingOrders(plugd::Dispatcher* dispatcher);
        virtual bool hasStampSpecification();
    };

    using namespace flowvr::xml;

    /// Constructor.
    MergeItExt::MergeItExt(const std::string objID)
      : Filter(objID), nbIn(1)
    {
    }

    MergeItExt::~MergeItExt()
    {
    }

    flowvr::plugd::Result MergeItExt::init(flowvr::xml::DOMElement* xmlRoot, flowvr::plugd::Dispatcher* dispatcher)
    {
      flowvr::plugd::Result result = Filter::init(xmlRoot, dispatcher);
      if (result.error()) return result;

      xml::DOMNodeList* lnbIn = xmlRoot->getElementsByTagName("nbIn");
      if (lnbIn->getLength()>=1)
      {
        nbIn = atoi(lnbIn->item(0)->getTextContent().c_str());
#ifdef __DEBUG
        std::cout << objectID() << ": nbIn=" << nbIn << std::endl;
#endif
        if (nbIn < 0) nbIn = 1;
      }
      delete lnbIn;

      initInputs(nbIn+1);
      //inputs[IDPORT_IN]->storeSpecification();
      inputs[IDPORT_ORDER]->setName("order");

      for (int i = 1; i <= nbIn; ++i)
      {
        char buf[10];
        sprintf(buf, "in%d", i-1);  // start counting at 0
#ifdef __DEBUG
        std::cout << objectID() << ": init port " << buf << std::endl;
#endif
        inputs[i]->setName(buf);
      }

      //only one outputmessagequeue for this filter
      initOutputs(1);
      outputs[0]->setName("out");
      //outputs[0]->msgtype=Message::FULL;  /TODO: test, see other todo

      return result;
    }

    void MergeItExt::newMessageNotification(int mqid, int msgnum, const Message& msg, Dispatcher* dispatcher)
    {
#ifdef __DEBUG
      if (mqid == IDPORT_ORDER)
        std::cout << objectID()<<": new order "<<msgnum<<" queue size "<<inputs[mqid]->size()<<std::endl;
      else
        std::cout << objectID()<<": new input "<<msgnum<<" queue size "<<inputs[mqid]->size()<<std::endl;
#endif
      sendPendingOrders(dispatcher);
    }

    void MergeItExt::newStampListSpecification(int mqid, const Message& msg, Dispatcher* dispatcher)
    {
      if (mqid != IDPORT_ORDER)
      {
        // forward specification to out port
#ifdef __DEBUG
        std::cout << objectID()<<": forwarding stamps specification"<<std::endl;
#endif

        //give the Stamplist to the outputmessage queue
        if (nbIn > 0)
        {
          outputs[0]->stamps = inputs[1]->getStampList();
          outputs[0]->newStampSpecification(dispatcher);
        }

        sendPendingOrders(dispatcher);
      }
    }

    bool MergeItExt::hasStampSpecification()
    {
      for (int i = 1; i <= nbIn; ++i)
      {
        if (!inputs[i]->stampsReceived())
        {
          printf("waiting for stamp specs");

          return false;
        }
      }
      return true;
    }

    void MergeItExt::sendPendingOrders(plugd::Dispatcher* dispatcher)
    { // MAIN FILTER FUNCTION

//for(;;){
      if (!hasStampSpecification()) return; // still waiting for stamps specification

      if (!inputs[IDPORT_ORDER]->frontMsg().valid())
      {
#ifdef __DEBUG
        std::cout << inputs[1]->frontMsg().valid() << std::endl;
        std::cout<<objectID()<<": waiting orders"<<std::endl;
#endif
        return;
      }

      inputs[IDPORT_ORDER]->eraseFront();

      MessagePut newmsg;
      bool hasStamps = false;

      // figure out size and set stamps
      size_t size = 0;
      for (int i = 1; i <= nbIn; ++i)
      {
        // TODO: iterate into queue too!
        const Message &msg = inputs[i]->frontMsg();
        if (msg.valid())
        {
          if (!hasStamps)
          {
            newmsg.stamps.clone(msg.stamps, &inputs[i]->getStampList());
            hasStamps = true;
          }
          size += msg.data.getSize();

      std::cout<<objectID()<<": found a message of "<< msg.data.getSize() << " bytes" << std::endl;
        }
      }


      // HACK: make sure to create a FULL message TODO: check if works without too...
      // by not specifying FULL ;)
      BufferWrite newdata = poolout.alloc(getAllocator(), size > 0 ? size : 1);

      // concatenate data.
      size_t pos = 0;
      for (int i = 1; i <= nbIn; ++i)
      {
        const Message &msg = inputs[i]->frontMsg();
        if (msg.valid()) {
          memcpy(newdata.writeAccess() + pos, msg.data.readAccess(), msg.data.getSize());
          pos += msg.data.getSize();
          inputs[i]->eraseFront();
        }
      }
      newmsg.data = newdata;




      //while(!inputs[IDPORT_IN]->empty())
      //{
      //if (inputs[IDPORT_IN]->frontMsg().valid())
      //inputs[IDPORT_IN]->eraseFront();
      //}

      /*
         for(unsigned i=first;i<msgs.size();i++)
         {
         int num=-2;
         int it=-2;
         msgs[i].stamps.read(inputs[IDPORT_IN]->getStampList().num,num);
         msgs[i].stamps.read(inputs[IDPORT_IN]->getStampList().it,it);
         std::string source;
         msgs[i].stamps.read(inputs[IDPORT_IN]->getStampList().source,source);
         std::cout << "MERGEIT"<<source<<": msgs["<<i<<"] num="<<num<<" it="<<it<<std::endl;
         }
         */
#ifdef __DEBUG
      std::cout<<objectID()<<": sending message "<<size<<" bytes"<<std::endl;
#endif
      outputs[0]->put(newmsg, dispatcher);
      newmsg.clear();
      newdata.clear();
    }
    //}

    flowvr::plugd::GenClass<MergeItExt> MergeItExtClass("flowvr.plugins.MergeItExt", // name
        "", // description
        &flowvr::plugins::FilterClass
        );

    Class* MergeItExt::getClass() const
    {
      return &MergeItExtClass;
    }

  } // namespace plugins

} // namespace flowvr
