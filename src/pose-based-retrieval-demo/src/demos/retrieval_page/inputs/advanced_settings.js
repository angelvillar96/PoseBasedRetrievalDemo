import React from "react"
import {Row, Col} from 'react-bootstrap'
import Collapsible from 'react-collapsible';

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

    var title = <span>{this.state.icon}Advanced Settings </span>

    return(
      <Collapsible trigger={title} className="advanced_settings"
                   onTriggerOpening={this.open}
                   onTriggerClosing={this.close}
                   open={false}>
        <Row>
          Hola
        </Row>
      </Collapsible>
    );
  }

}

export default AdvancedSettings
