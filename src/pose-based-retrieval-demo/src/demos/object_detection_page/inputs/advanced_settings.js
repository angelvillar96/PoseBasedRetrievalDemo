  import React from "react"
import {Container, Row, Col} from 'react-bootstrap'
import Collapsible from 'react-collapsible';

import CustomDropdown from "../../../components/dropdown/dropdown.js"
import {BsPlusCircle} from "react-icons/bs"
import {AiOutlineMinusCircle} from "react-icons/ai"

import "./styles/input_styles.css"

class AdvancedSettings extends React.Component{


  constructor(props){
    super(props)
    this.state = {
      icon: <BsPlusCircle/>  // if True, expand is closed
    }

    this.open = this.open.bind(this)
    this.close = this.close.bind(this)
  }

  open(){
    this.setState({icon: <AiOutlineMinusCircle/>})
    return
  }
  close(){
    this.setState({icon: <BsPlusCircle/>})
  }

  render(){

    var title = <span>{this.state.icon}  Advanced Settings </span>

    const object_detectors = ["Tuned R-CNN", "Faster R-CNN"]
    const retrieval_databases = ["Art History", "MS-COCO"] // "Christian Arch", "Classical Arch"

    return(
      <Collapsible trigger={title} className="advanced_settings"
                   onTriggerOpening={this.open}
                   onTriggerClosing={this.close}
                   open={false}>
        <div className="expandable_area">
          <Row className="expandable_row">
            <Col sm={1}></Col>
            <Col sm={4}>
              <CustomDropdown name="Object Detector" id="object_detector"
                              options={object_detectors}
                              update_state={this.props.update_state}/>
            </Col>
            <Col sm={1}></Col>
            <Col sm={4}>
              <CustomDropdown name="Database" id="retrieval_database"
                              options={retrieval_databases}
                              update_state={this.props.update_state}/>
            </Col>
          </Row>
        </div>
      </Collapsible>
    );
  }

}

export default AdvancedSettings
