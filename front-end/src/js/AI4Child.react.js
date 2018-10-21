import React from 'react';
import ReactDOM from 'react-dom';

class AI4Child extends React.Component {
    constructor() {
        super();

        this.state = {
        };

        this.videoRef = React.createRef();

//        this.handleChange = this.handleChange.bind(this);
    }

    componentDidMount() {
        console.log(this.videoRef, this.props);
        this.videoRef.srcObject = this.props.stream;
    }
    
    render() {
        return (
            <div>
                <video ref={this.videoRef} autoplay="true" />
            </div>
        );
        //srcObject={this.props.stream} autoplay={true} />;
    }
}

export default AI4Child;

